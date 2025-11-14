#!/bin/bash

# WRF-GSplus 모델 테스트 스크립트
# 사용법: ./run_test.sh [체크포인트_파일] [데이터셋_경로]

# 기본값 설정
MODEL_PATH="./output/20251017_074111"
CHECKPOINT_FILE=""
DATASET_PATH="./data_test200"
OUTPUT_PATH="./test_results"

# 명령행 인수 처리
if [ $# -ge 1 ]; then
    CHECKPOINT_FILE="$1"
else
    echo "사용법: $0 <체크포인트_파일> [데이터셋_경로] [출력_경로]"
    echo "예시: $0 ./output/20251017_074111/chkpnt200000.pth ./data_test200 ./test_results"
    exit 1
fi

if [ $# -ge 2 ]; then
    DATASET_PATH="$2"
fi

if [ $# -ge 3 ]; then
    OUTPUT_PATH="$3"
fi

# 체크포인트 파일 존재 확인
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_FILE"
    exit 1
fi

# 데이터셋 경로 존재 확인
if [ ! -d "$DATASET_PATH" ]; then
    echo "오류: 데이터셋 경로를 찾을 수 없습니다: $DATASET_PATH"
    exit 1
fi

echo "=== WRF-GSplus 모델 테스트 ==="
echo "모델 경로: $MODEL_PATH"
echo "체크포인트 파일: $CHECKPOINT_FILE"
echo "데이터셋 경로: $DATASET_PATH"
echo "출력 경로: $OUTPUT_PATH"
echo "================================"

# Python 환경 활성화 (필요한 경우)
# source activate your_environment

# 테스트 실행
python test.py \
    --model_path "$MODEL_PATH" \
    --checkpoint "$CHECKPOINT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --gpu 0

echo "테스트 완료! 결과는 $OUTPUT_PATH 에서 확인하세요."
