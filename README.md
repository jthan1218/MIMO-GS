# WRF-GS+
Welcome to the WRF-GS+ repository, an enhanced version of the original WRF-GS. You can access the accompanying paper, ["Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach"](https://arxiv.org/abs/2412.04832v4), for more details.

Abstract: Wireless channel modeling plays a pivotal role in designing, analyzing, and optimizing wireless communication systems. Nevertheless, developing an effective channel modeling approach has been a long-standing challenge. This issue has been escalated due to denser network deployment, larger antenna arrays, and broader bandwidth in next-generation networks. To address this challenge, we put forth WRF-GS, a novel framework for channel modeling based on wireless radiation field (WRF) reconstruction using 3D Gaussian splatting (3D-GS). WRF-GS employs 3D Gaussian primitives and neural networks to capture the interactions between the environment and radio signals, enabling efficient WRF reconstruction and visualization of the propagation characteristics. The reconstructed WRF can then be used to synthesize the spatial spectrum for comprehensive wireless channel characterization. While WRF-GS demonstrates remarkable effectiveness, it faces limitations in capturing high-frequency signal variations caused by complex multipath effects. To overcome these limitations, we propose WRF-GS+, an enhanced framework that integrates electromagnetic wave physics into the neural network design. WRF-GS+ leverages deformable 3D Gaussians to model both static and dynamic components of the WRF, significantly improving its ability to characterize signal variations. In addition, WRF-GS+ enhances the splatting process by simplifying the 3D-GS modeling process and improving computational efficiency. Experimental results demonstrate that both WRF-GS and WRF-GS+ outperform baselines for spatial spectrum synthesis, including ray tracing and other deep-learning approaches.

## Installation
Create the basic environment
```python
conda env create --file environment.yml
conda activate wrfgsplus
```
Install some extensions
```python
cd submodules
pip install ./simple-knn
pip install ./diff-gaussian-rasterization # or cd ./diff-gaussian-rasterization && python setup.py develop
pip install ./fused-ssim
```

## Training & Evaluation

### Training
Due to file size limitations, a small dataset is included to help quickly verify the code, which can be executed using the following command:
```bash
python train.py
```
More datasets can be found [here](https://github.com/XPengZhao/NeRF2?tab=readme-ov-file).

### Testing Trained Models
훈련된 모델을 테스트하려면 다음 방법을 사용할 수 있습니다:

#### 방법 1: Python 스크립트 직접 실행
```bash
python test.py --model_path ./output/20251017_074111 --checkpoint ./output/20251017_074111/chkpnt200000.pth --dataset_path ./data_test200 --output_path ./test_results
```

#### 방법 2: 쉘 스크립트 사용 (권장)
```bash
./run_test.sh ./output/20251017_074111/chkpnt200000.pth ./data_test200 ./test_results
```

#### 사용 가능한 체크포인트 파일들:
- `chkpnt7000.pth` - 7,000번째 iteration
- `chkpnt30000.pth` - 30,000번째 iteration  
- `chkpnt60000.pth` - 60,000번째 iteration
- `chkpnt200000.pth` - 200,000번째 iteration (최종)

#### 테스트 결과:
테스트 실행 후 다음 파일들이 생성됩니다:
- `test_results.txt` - 전체 결과 요약
- `all_ssim.txt` - 각 샘플별 SSIM 값
- `pred_spectrum/` - 예측 결과와 실제 결과 비교 이미지들
- `test_logger.log` - 상세한 로그 파일

## To-Do List
- [ ] Release more case study code.
- [ ] Optimize related code structure.

## BibTex
If you find this work useful in your research, please cite:
```bibtex
@article{wen2025wrfgsplus,
  title={Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach},
  author={Wen, Chaozheng and Tong, Jingwen and Hu, Yingdong and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2412.04832v3},
  year={2025}
}
```
## Acknowledgment
Some code snippets are borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/main).
