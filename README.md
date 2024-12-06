# Arabic Character Recognition App 🖋️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arabic-character-recognition.streamlit.app)

An interactive web application that recognizes handwritten Arabic characters using deep learning. Draw an Arabic character, and the model will predict what it is!

## 🌟 Features

- **Real-time Recognition**: Instantly recognize handwritten Arabic characters
- **Interactive Canvas**: User-friendly drawing interface
- **Top 3 Predictions**: Shows the top three most likely characters with confidence scores
- **Processed Image Display**: See how your drawing is processed before prediction

## 🚀 Try It Out

Visit the live app: [Arabic Character Recognition](https://arabic-character-recognition.streamlit.app)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow
- **Drawing Interface**: Streamlit-drawable-canvas
- **Language**: Python

## 📝 How to Use

1. Visit the [web application](https://arabic-character-recognition.streamlit.app)
2. Draw an Arabic character in the canvas
3. Click 'Predict'
4. View the top 3 predictions and their confidence scores

## 🧮 Supported Characters

The model can recognize 28 Arabic characters:
ي ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و

## 🔧 Local Development

To run this project locally:

1. Clone the repository
git clone https://github.com/rakanalb/arabic-character-recognition.git

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

## 📊 Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: 32x32 grayscale images
- Output: 28 Arabic character classes
- Training Dataset: Arabic Handwritten Characters Dataset

## 👥 Contributing

Feel free to open issues and pull requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit for the amazing web app framework
- TensorFlow team for the deep learning tools
- Arabic Handwritten Characters Dataset creators

---
Developed with ❤️ & tears by Rakan Albadeen & Thamer Algahtani
