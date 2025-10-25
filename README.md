
# K-Market Research Engine v0.3 (Ticker+Theme → Predict)
- Streamlit에서 **종목 코드 + 테마 강도만 입력**하면:
  1) Yahoo Finance에서 가격/거시 수집(가능 시)
  2) 특징 자동 생성
  3) 모델 자동 학습(선택)
  4) 예측 결과(prob_up, signal) 출력

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
## Deploy on Streamlit Cloud
Main file: `streamlit_app.py`
