# 🎙️ Deepfake Voice Detection

An end-to-end AI system to detect deepfake voice recordings using both **audio features** and **transcribed text** from the audio. 
This project uses **XGBoost classifiers** trained on engineered features and **OpenAI Whisper** for transcription.

## 🚀 Features

- 🎧 Extracts 43 audio-based features (MFCCs, pitch variance, spectral contrast, etc.)
- 📝 Generates transcripts using Whisper and processes text via TF-IDF
- 🤖 Trains three models: audio-only, text-only, and combined
- 📊 Evaluates models with confusion matrix, ROC-AUC, and classification report
- 💡 Provides **explainable output** to understand prediction reasoning
- 🌐 Includes an interactive **Streamlit web app** for real-time prediction

## 🧠 Tech Stack

- **Python**, **Librosa**, **Whisper**
- **XGBoost**, **scikit-learn**
- **Streamlit** for UI
- **Pandas**, **TQDM**, **Joblib**

## 📂 Project Structure

📦 Deepfake Voice Detection
├── data_preparation.py
├── feature_extraction.py
├── vectorize_features.py
├── train_xgboost.py
├── app.py


## 📈 Model Results

| Model        | Accuracy | AUC     |
|--------------|----------|---------|
| Audio-only   | ~84%     | ~0.91   |
| Text-only    | ~56%     | ~0.58   |
| Combined     | **~80%** | **0.87** |

## 🔬 How It Works

1. **Audio Feature Extraction**: MFCCs, pitch, spectral contrast, etc.
2. **Text Transcript Generation**: Uses Whisper model to transcribe.
3. **Vectorization**: TF-IDF for text + StandardScaler for audio.
4. **Model Training**: Trains XGBoost on audio, text, and combined inputs.
5. **Prediction**: Final decision made using combined model and fallback logic.
6. **Explanation**: Interpretable reasons provided for every prediction.


