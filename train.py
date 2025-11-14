#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import datetime
import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.generate_camera import generate_new_cam

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from scipy.spatial.transform import Rotation
from utils.data_painter import paint_spectrum_compare 
from skimage.metrics import structural_similarity as ssim





def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    datadir = 'data'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    devices = torch.device('cuda')
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,current_time)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.gaussian_init()
    deform = DeformModel()
    deform.train_setting(opt)
    
    scene = Scene(dataset, gaussians)
    scene.dataset_init()
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = None
    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # Get dataset size for progress bar
    dataset_size = len(scene.train_set)
    # Set iterations to dataset_size (1 epoch)
    opt.iterations = dataset_size
    total_epochs = 1
    
    # Use opt.iterations as testing_iterations
    testing_iterations = [opt.iterations]
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 1000 == 0:
            print("nums of gaussians:", gaussians.get_xyz.shape[0])

        # Pick a random Camera
        try:
            channels_beam, rx_pos = next(scene.train_iter_dataset)

        except:
            scene.dataset_init()
            channels_beam, rx_pos = next(scene.train_iter_dataset)

        r_o = scene.r_o
        gateway_orientation = scene.gateway_orientation 
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        # Remove batch dimension since batch_size=1
        rx_pos = rx_pos.squeeze(0).cuda()
        viewpoint_cam = generate_new_cam(R, r_o)
        N = gaussians.get_xyz.shape[0]
        time_input = rx_pos.expand(N, -1)
        d_xyz, d_rotation, d_scaling, d_signal = deform.step(gaussians.get_xyz.detach(), time_input)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # if iteration%100==0:
        #     print("radii.shape:", radii.shape)
        #     print('radii:', radii)
        
        channel, height, width = image.shape
        # Reshape render output to match channels_beam shape (2, 4, 16)
        # image is (2, height, width), we need (2, 4, 16)
        # Assuming height*width >= 64, we take the first 64 elements and reshape
        total_elements = height * width
        if total_elements >= 64:
            # Flatten and take first 64 elements, then reshape to (2, 4, 16)
            pred_channels_beam = image.reshape(2, -1)[:, :64].reshape(2, 4, 16)
        else:
            # If not enough elements, pad with zeros
            pred_channels_beam = torch.zeros(2, 4, 16, device=image.device, dtype=image.dtype)
            pred_channels_beam[:, :height, :width] = image
        
        # For visualization, show first channel
        render_image_show = pred_channels_beam[0].unsqueeze(0).cuda()
        tb_writer.add_image('render-img', render_image_show, iteration)
        
        # Loss: Real MSE + Imaginary MSE
        # channels_beam is (batch_size, 2, 4, 16) where [:, 0] is real and [:, 1] is imaginary
        # Remove batch dimension since batch_size=1
        gt_channels_beam = channels_beam.squeeze(0).cuda()  # (2, 4, 16)
        
        # Separate real and imaginary parts
        pred_real = pred_channels_beam[0]  # (4, 16)
        pred_imag = pred_channels_beam[1]  # (4, 16)
        gt_real = gt_channels_beam[0]  # (4, 16)
        gt_imag = gt_channels_beam[1]  # (4, 16)
        
        # MSE for real and imaginary parts
        mse_real = torch.mean((pred_real - gt_real) ** 2)
        mse_imag = torch.mean((pred_imag - gt_imag) ** 2)
        
        loss = mse_real + mse_imag

        # Depth regularization
        Ll1depth_pure = 0.0
        
        Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                # Calculate current epoch and position in dataset
                current_epoch = (iteration - 1) // dataset_size + 1
                position_in_epoch = ((iteration - 1) % dataset_size) + 1
                progress_bar.set_postfix({
                    "Epoch": f"{current_epoch}/{total_epochs}",
                    "Sample": f"{position_in_epoch}/{dataset_size}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "MSE Real": f"{mse_real.item():.{7}f}",
                    "MSE Imag": f"{mse_imag.item():.{7}f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            tb_writer.add_scalar('train_loss', loss.item(), iteration)
            tb_writer.add_scalar('train_loss/mse_real', mse_real.item(), iteration)
            tb_writer.add_scalar('train_loss/mse_imag', mse_imag.item(), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # Deform model also saved
                deform.save_weights(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
            


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
           
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                
                print("Start evaluation")
                # Save test results to output folder
                iteration_path = os.path.join(scene.model_path, 'pred_channels_beam', str(iteration))
                os.makedirs(iteration_path, exist_ok=True) 
                full_path = os.path.join(scene.model_path, str(iteration))
                os.makedirs(full_path, exist_ok=True)
                
                # Randomly select 50 test samples
                test_dataset_size = len(scene.test_set)
                num_test_samples = min(50, test_dataset_size)
                test_indices = np.random.choice(test_dataset_size, num_test_samples, replace=False)
                print(f"Testing on {num_test_samples} randomly selected samples out of {test_dataset_size} total test samples")
                
                save_img_idx = 0
                all_mse_real = []
                all_mse_imag = []
                for idx in test_indices:
                    # Get test sample by index
                    test_channels_beam, test_rx_pos = scene.test_set[idx]
                    test_channels_beam = test_channels_beam.unsqueeze(0)  # Add batch dimension
                    test_rx_pos = test_rx_pos.unsqueeze(0)  # Add batch dimension 
                    
                    
                    r_o = scene.r_o
                    gateway_orientation = scene.gateway_orientation 
                    R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
                    # Remove batch dimension since batch_size=1
                    rx_pos = test_rx_pos.squeeze(0).cuda()
                    viewpoint_cam = generate_new_cam(R, r_o)
                    N = gaussians.get_xyz.shape[0]
                    time_input = rx_pos.expand(N, -1)
                    d_xyz, d_rotation, d_scaling, d_signal = deform.step(gaussians.get_xyz.detach(), time_input)
                    
                    
            

                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal)

                    image = render_pkg["render"]
                    channel, height, width = image.shape
                    
                    # Reshape render output to match channels_beam shape (2, 4, 16)
                    total_elements = height * width
                    if total_elements >= 64:
                        pred_channels_beam = image.reshape(2, -1)[:, :64].reshape(2, 4, 16)
                    else:
                        pred_channels_beam = torch.zeros(2, 4, 16, device=image.device, dtype=image.dtype)
                        pred_channels_beam[:, :height, :width] = image

                    # Convert to numpy for evaluation
                    pred_channels_beam_np = pred_channels_beam.detach().cpu().numpy()
                    gt_channels_beam_np = test_channels_beam.squeeze(0).detach().cpu().numpy()
                    
                    # Calculate MSE for real and imaginary parts
                    pred_real = pred_channels_beam_np[0]
                    pred_imag = pred_channels_beam_np[1]
                    gt_real = gt_channels_beam_np[0]
                    gt_imag = gt_channels_beam_np[1]
                    
                    mse_real = np.mean((pred_real - gt_real) ** 2)
                    mse_imag = np.mean((pred_imag - gt_imag) ** 2)
                    
                    all_mse_real.append(mse_real)
                    all_mse_imag.append(mse_imag)
                    
                    print(
                        "Sample {:d} (idx {:d}), MSE Real = {:.6f}; MSE Imag = {:.6f}".format(save_img_idx, idx, mse_real, mse_imag))
                    
                    # Convert to complex arrays (4, 16) complex64
                    pred_channels_beam_complex = pred_real + 1j * pred_imag
                    
                    # Get original gt_channels_beam directly from test_normalized.mat to ensure exact match
                    import scipy.io
                    test_mat_path = os.path.join(scene.datadir, 'test_normalized.mat')
                    test_mat_data = scipy.io.loadmat(test_mat_path)
                    gt_channels_beam_complex = test_mat_data['channels_beam'][idx].astype(np.complex64)
                    
                    # Save as .mat file
                    scipy.io.savemat(os.path.join(iteration_path, f'{save_img_idx}.mat'),
                                    {'pred_channels_beam': pred_channels_beam_complex.astype(np.complex64),
                                     'gt_channels_beam': gt_channels_beam_complex})
                    
                    print("Mean MSE Real: {:.6f}, Mean MSE Imag: {:.6f}".format(
                        np.mean(all_mse_real), np.mean(all_mse_imag)))
                    save_img_idx += 1
                    np.savetxt(os.path.join(full_path, 'all_mse_real.txt'), all_mse_real, fmt='%.6f')
                    np.savetxt(os.path.join(full_path, 'all_mse_imag.txt'), all_mse_imag, fmt='%.6f')

                torch.cuda.empty_cache() 

    # 학습 완료 후 최종 모델 저장
    print("\n[FINAL] Saving final model")
    scene.save(opt.iterations)
    deform.save_weights(scene.model_path, opt.iterations)
    # 최종 checkpoint도 저장
    torch.save((gaussians.capture(), opt.iterations), scene.model_path + "/chkpnt_final.pth")
    print("Final model saved to: {}".format(scene.model_path))



def prepare_output_and_logger(args,time):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.74")
    parser.add_argument('--port', type=int, default=6074)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000,60000,200000,300000,600000,1200000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000, 60000, 200000, 300000, 600000,1200000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args(sys.argv[1:])
    
    args.save_iterations.append(args.iterations)
    torch.cuda.set_device(args.gpu)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
