# 🧱 토압 계산 프로그램 (Earth Pressure Calculator)

Rankine & Coulomb 토압 이론 기반 옹벽 설계용 토압 계산기 (Streamlit 웹앱)

## ✨ 주요 기능
- **랭킨(Rankine) 주동·수동 토압 계수 Ka, Kp 계산** (수평/경사 지표면 모두 지원)
- **쿨롱(Coulomb) 주동·수동 토압 계수 계산** (벽면마찰각 δ, 옹벽 배면 각도 α 포함)
- **토압 합력(P)과 작용점(ȳ) 자동 산정**
- **점착력 c, 등분포하중 q, 지하수위 고려** 옵션
- **토압 분포도 그래프 + 옹벽 단면 개념도 자동 작도**
- 결과를 Rankine vs Coulomb 비교 표로 한눈에 확인

## 📦 설치 및 실행 (로컬)

```bash
# 1) 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 실행
streamlit run app.py
```

실행하면 자동으로 브라우저(`http://localhost:8501`)가 열립니다.

## 🌐 웹 배포 (무료)

### Streamlit Community Cloud
1. GitHub에 본 폴더(`app.py`, `requirements.txt`)를 푸시
2. https://streamlit.io/cloud 접속 → GitHub 계정 연결
3. **New app** → 리포지토리/브랜치/`app.py` 선택 → Deploy
4. 끝! `https://<your-app>.streamlit.app` 으로 누구나 접속 가능

### 그 외 옵션
- **Hugging Face Spaces**: SDK를 Streamlit으로 선택해서 배포
- **Render / Railway**: Dockerfile 없이도 Procfile만 추가하면 배포 가능

## 📥 입력 항목
| 항목 | 단위 | 비고 |
|---|---|---|
| 단위중량 γ | t/m³ | 흙의 습윤단위중량 |
| 내부마찰각 φ | ° | |
| 점착력 c | t/m² | 0이면 사질토 |
| 옹벽 높이 H | m | |
| 지표면 경사각 i | ° | 수평이면 0 |
| 옹벽 배면각 α | ° | 수직벽이면 90 |
| 벽면마찰각 δ | ° | 쿨롱용, 보통 (1/2~2/3)φ |
| 등분포하중 q | t/m² | 선택 |
| 지하수위 깊이 | m | 선택 |

## 📐 사용된 공식
- **Rankine (수평)**: `Ka = tan²(45° − φ/2)`, `Kp = tan²(45° + φ/2)`
- **Rankine (경사 i)**: `Ka = cos i · (cos i − √(cos²i − cos²φ)) / (cos i + √(cos²i − cos²φ))`
- **Coulomb**: 표준 4-각도식(α, β, φ, δ) — 코드 내 `coulomb_coefficients` 함수 참조
- **합력**: `P = ∫₀ᴴ σ(z) dz` (사다리꼴 적분)
- **작용점**: `ȳ = ∫σ(z)·(H−z)dz / ∫σ(z)dz`

## ⚠️ 가정 / 한계
- 점착력 `-2c√Ka`는 주동에서만 반영, 인장영역(σ<0)은 0으로 절단
- 지하수위 아래에서는 수중단위중량 `γ' = γ − γw` 사용 + 정수압 별도 가산
- 수동토압의 c, q 효과는 보수적으로 미반영(일반 설계 관행)
- 본 프로그램은 학습/개략설계용, 실무 설계는 **KDS / KSCE** 기준 및 안전율 적용 필수
