#!/usr/bin/env python3
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

import os
import torch
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel, DeformModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.generate_camera import generate_new_cam
from scipy.spatial.transform import Rotation
from utils.data_painter import paint_spectrum_compare
from skimage.metrics import structural_similarity as ssim
from utils.logger import logger_config
import datetime

def test_model(model_path, checkpoint_path, dataset_path, output_path=None):
    """
    훈련된 모델을 로드하고 테스트 데이터셋에 대해 평가를 수행합니다.
    
    Args:
        model_path: 훈련된 모델이 저장된 경로
        checkpoint_path: 체크포인트 파일 경로 (.pth 파일)
        dataset_path: 테스트 데이터셋 경로
        output_path: 결과를 저장할 경로 (선택사항)
    """
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 출력 경로 설정
    if output_path is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('test_results', current_time)
    
    os.makedirs(output_path, exist_ok=True)
    
    # 로거 설정
    log_filename = "test_logger.log"
    log_savepath = os.path.join(output_path, log_filename)
    logger = logger_config(log_savepath=log_savepath, logging_name='test')
    logger.info("Starting model testing")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output path: {output_path}")
    
    # 모델 파라미터 설정
    parser = ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser)
    pipe_params = PipelineParams(parser)
    opt_params = OptimizationParams(parser)
    
    # 데이터셋 경로 설정
    model_params.source_path = dataset_path
    model_params.model_path = model_path
    model_params.data_device = "cuda"
    model_params.eval = True
    
    # Gaussian 모델 초기화
    gaussians = GaussianModel(model_params.sh_degree)
    gaussians.gaussian_init()
    
    # Scene 초기화
    scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)
    scene.dataset_init()
    
    # Deform 모델 초기화
    deform = DeformModel()
    deform.train_setting(opt_params)
    
    # 체크포인트 로드
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gaussians_params, iteration = checkpoint
    gaussians.restore(gaussians_params, opt_params)
    
    # Deform 모델 가중치 로드 (선택사항)
    try:
        logger.info(f"Loading deform model weights for iteration {iteration}")
        deform.load_weights(model_path, iteration)
        logger.info("Deform model weights loaded successfully")
    except Exception as e:
        logger.info(f"Deform model weights not found (this is normal for some checkpoints)")
        logger.info("Continuing with basic Gaussian model")
    
    # 배경 설정
    bg_color = [1, 1, 1] if model_params._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    
    # 테스트 실행
    logger.info("Starting evaluation on test dataset")
    
    all_ssim = []
    all_l1_errors = []
    save_img_idx = 0
    
    # 테스트 결과 저장 디렉토리 생성
    pred_spectrum_path = os.path.join(output_path, 'pred_channels_beam')
    os.makedirs(pred_spectrum_path, exist_ok=True)
    
    # Randomly select 50 test samples
    test_dataset_size = len(scene.test_set)
    num_test_samples = min(50, test_dataset_size)
    test_indices = np.random.choice(test_dataset_size, num_test_samples, replace=False)
    logger.info(f"Testing on {num_test_samples} randomly selected samples out of {test_dataset_size} total test samples")
    
    with torch.no_grad():
        for save_img_idx, idx in enumerate(test_indices):
            # Get test sample by index
            test_channels_beam, test_rx_pos = scene.test_set[idx]
            test_channels_beam = test_channels_beam.unsqueeze(0)  # Add batch dimension
            test_rx_pos = test_rx_pos.unsqueeze(0)  # Add batch dimension
            logger.info(f"Processing test sample {save_img_idx}")
            
            # 카메라 설정
            r_o = scene.r_o
            gateway_orientation = scene.gateway_orientation 
            R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
            # Remove batch dimension since batch_size=1
            rx_pos = test_rx_pos.squeeze(0).cuda()
            viewpoint_cam = generate_new_cam(R, r_o)
            
            # Deform 적용 (가중치 유무와 관계없이 안전한 기본값 준비)
            N = gaussians.get_xyz.shape[0]
            time_input = rx_pos.expand(N, -1)
            d_xyz = torch.zeros_like(gaussians.get_xyz)
            d_rotation = torch.zeros_like(gaussians.get_rotation)
            d_scaling = torch.zeros_like(gaussians.get_scaling)
            d_signal = torch.zeros((N, 3), device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
            try:
                d_xyz_tmp, d_rotation_tmp, d_scaling_tmp, d_signal_tmp = deform.step(gaussians.get_xyz.detach(), time_input)
                d_xyz, d_rotation, d_scaling, d_signal = d_xyz_tmp, d_rotation_tmp, d_scaling_tmp, d_signal_tmp
            except Exception as e:
                logger.info(f"Deform step failed, using zeros. Reason: {str(e)}")
            render_pkg = render(viewpoint_cam, gaussians, pipe_params, background, d_xyz, d_rotation, d_scaling, d_signal)
            
            image = render_pkg["render"]
            channel, height, width = image.shape
            
            # Reshape render output to match channels_beam shape (2, 4, 16)
            total_elements = height * width
            if total_elements >= 64:
                pred_channels_beam = image.reshape(2, -1)[:, :64].reshape(2, 4, 16)
            else:
                pred_channels_beam = torch.zeros(2, 4, 16, device=image.device, dtype=image.dtype)
                pred_channels_beam[:, :height, :width] = image
            
            # 예측 결과를 numpy로 변환
            # Remove batch dimension since batch_size=1
            pred_channels_beam_np = pred_channels_beam.detach().cpu().numpy()
            gt_channels_beam_np = test_channels_beam.squeeze(0).detach().cpu().numpy()
            
            # Calculate MSE for real and imaginary parts
            pred_real = pred_channels_beam_np[0]
            pred_imag = pred_channels_beam_np[1]
            gt_real = gt_channels_beam_np[0]
            gt_imag = gt_channels_beam_np[1]
            
            mse_real = np.mean((pred_real - gt_real) ** 2)
            mse_imag = np.mean((pred_imag - gt_imag) ** 2)
            total_mse = mse_real + mse_imag
            
            all_ssim.append(total_mse)  # Using total MSE as metric
            all_l1_errors.append(np.mean(np.abs(pred_real - gt_real)) + np.mean(np.abs(pred_imag - gt_imag)))
            
            logger.info(f"Sample {save_img_idx} (idx {idx}): MSE Real = {mse_real:.6f}, MSE Imag = {mse_imag:.6f}, Total MSE = {total_mse:.6f}")
            
            # Convert to complex arrays (4, 16) complex64
            pred_channels_beam_complex = pred_real + 1j * pred_imag
            
            # Get original gt_channels_beam directly from test_normalized.mat to ensure exact match
            import scipy.io
            test_mat_path = os.path.join(scene.datadir, 'test_normalized.mat')
            test_mat_data = scipy.io.loadmat(test_mat_path)
            gt_channels_beam_complex = test_mat_data['channels_beam'][idx].astype(np.complex64)
            
            # Save as .mat file
            scipy.io.savemat(os.path.join(pred_spectrum_path, f'{save_img_idx}.mat'),
                            {'pred_channels_beam': pred_channels_beam_complex.astype(np.complex64),
                             'gt_channels_beam': gt_channels_beam_complex})
    
    # 전체 결과 요약
    mean_mse = np.mean(all_ssim)  # Actually total MSE
    median_mse = np.median(all_ssim)
    mean_l1 = np.mean(all_l1_errors)
    
    logger.info(f"Test completed!")
    logger.info(f"Total samples: {len(all_ssim)}")
    logger.info(f"Mean Total MSE: {mean_mse:.6f}")
    logger.info(f"Median Total MSE: {median_mse:.6f}")
    logger.info(f"Mean L1 Error: {mean_l1:.6f}")
    
    # 결과를 파일로 저장
    results_file = os.path.join(output_path, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Results Summary\n")
        f.write(f"===================\n")
        f.write(f"Total samples: {len(all_ssim)}\n")
        f.write(f"Mean Total MSE: {mean_mse:.6f}\n")
        f.write(f"Median Total MSE: {median_mse:.6f}\n")
        f.write(f"Mean L1 Error: {mean_l1:.6f}\n")
        f.write(f"\nPer-sample Total MSE values:\n")
        for i, mse_val in enumerate(all_ssim):
            f.write(f"Sample {i}: {mse_val:.6f}\n")
    
    # MSE 값들을 별도 파일로 저장
    mse_file = os.path.join(output_path, 'all_total_mse.txt')
    np.savetxt(mse_file, all_ssim, fmt='%.6f')
    
    print(f"\nTest completed! Results saved to: {output_path}")
    print(f"Mean Total MSE: {mean_mse:.6f}")
    print(f"Median Total MSE: {median_mse:.6f}")
    print(f"Mean L1 Error: {mean_l1:.6f}")

def main():
    parser = ArgumentParser(description="Test trained WRF-GSplus model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the checkpoint file (.pth)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the test dataset")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save test results (optional)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device to use")
    
    args = parser.parse_args()
    
    # GPU 설정
    torch.cuda.set_device(args.gpu)
    
    # 시스템 상태 초기화
    safe_state(False)
    
    # 테스트 실행
    test_model(args.model_path, args.checkpoint, args.dataset_path, args.output_path)

if __name__ == "__main__":
    main()
