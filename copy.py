import os
import shutil

# ───────────────────────────────────────────────────────────────────────────────
# 1) 원본(소스) 루트 디렉토리와, 새롭게 val_이미지만 모아둘 대상 디렉토리를 정의
#    (자신의 환경에 맞게 경로만 수정하세요)
# ───────────────────────────────────────────────────────────────────────────────
SRC_ROOT = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\diffuseMix-main\\finetuned_model"
DST_ROOT = "C:\\Users\\Seo\\Desktop\\Gits\\ProjectPractice\\diffuseMix-main\\val_images_only"


# ───────────────────────────────────────────────────────────────────────────────
# 2) os.walk로 SRC_ROOT 밑을 재귀적으로 순회하면서,
#    하위 폴더(path) 내의 파일 리스트 중 val_*.png 파일만 골라낸 뒤,
#    DST_ROOT/원본_상대경로 폴더를 생성 후 파일 복사
# ───────────────────────────────────────────────────────────────────────────────
for root, dirs, files in os.walk(SRC_ROOT):
    # root: 현재 탐색 중인 폴더 전체 경로
    # files: 해당 폴더 내 존재하는 파일명 리스트
    
    # (1) SRC_ROOT로부터의 상대 경로(rel_path)를 계산
    # 예를 들어 root가 
    #   C:\...\finetuned_model\checkpoint-epoch3
    # 라면 rel_path는 "checkpoint-epoch3"가 된다.
    rel_path = os.path.relpath(root, SRC_ROOT)
    
    # (2) 현재 폴더(files) 중에서 'val_'로 시작하고 '.png'로 끝나는 파일만 필터링
    val_png_list = [
        fn for fn in files
        if fn.startswith("val_") and fn.lower().endswith(".png")
    ]
    
    # (3) 만약 val_*.png 파일이 하나라도 있다면, 복사할 대상 디렉토리를 만든 뒤 복사
    if len(val_png_list) > 0:
        # 3-1) 원본의 상대경로를 그대로 유지한 채 DST_ROOT 아래에 새 폴더 생성
        # 예: DST_ROOT\checkpoint-epoch3
        dst_dir = os.path.join(DST_ROOT, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 3-2) val_이미지 파일을 하나씩 복사
        for fn in val_png_list:
            src_file = os.path.join(root, fn)
            dst_file = os.path.join(dst_dir, fn)
            # shutil.copy2를 쓰면 메타데이터(최종 수정 시각 등)도 함께 복사됩니다.
            shutil.copy2(src_file, dst_file)

        print(f"[복사 완료] {root} 에서 {len(val_png_list)}개 파일 → {dst_dir}")

print(">>> 작업이 모두 완료되었습니다.")