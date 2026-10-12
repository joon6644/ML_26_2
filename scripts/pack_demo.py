"""시연·예측에 필요한, git에 없는 파일(모델·모델 입력 데이터)을 zip 하나로 묶는다.

    python scripts/pack_demo.py          # 예측 시연용 (models/ + model_input) → demo_bundle.zip
    python scripts/pack_demo.py --eda    # + EDA 노트북 재실행용 dataset·features·price

다른 컴퓨터에서는 git clone 한 프로젝트 폴더 맨 위에 압축을 풀면 원래 위치로 들어간다.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ("agri", "livestock", "fishery")
files = sorted((ROOT / "models").glob("*.joblib"))
for d in DOMAINS:
    files += [ROOT / "data" / "processed" / d / n for n in ("model_input.parquet", "model_input.json")]
    if "--eda" in sys.argv:
        files += [ROOT / "data" / "processed" / d / n for n in ("dataset.parquet", "features.parquet", "price.parquet")]

missing = [f for f in files if not f.exists()]
if missing:
    sys.exit("없는 파일: " + ", ".join(str(f.relative_to(ROOT)) for f in missing))
out = ROOT / "demo_bundle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # parquet은 이미 압축돼 있어 다시 압축하지 않음
    for f in files:
        z.write(f, f.relative_to(ROOT).as_posix())
print(f"{out.name}  {len(files)}개 파일, {out.stat().st_size / 1e6:,.0f} MB → 구글 드라이브에 올리고, 다른 컴퓨터의 프로젝트 폴더 맨 위에서 압축 해제")
