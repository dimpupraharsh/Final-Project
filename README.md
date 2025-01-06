# Image Captioning with CNN and BiLSTM Decoder

## 📌 **Project Overview**  
This project aims to develop an advanced image captioning system by combining **Convolutional Neural Networks (CNNs)** for image feature extraction with **Bidirectional Long Short-Term Memory (BiLSTM)** networks for caption generation. The project leverages various **pre-trained CNN models** as encoders, coupled with **BiLSTM decoders** enhanced by attention mechanisms to improve caption quality and contextual alignment.

## 🛠️ **Technical Details**  
### **Dataset:**  
- **Flickr 8k Dataset** – A widely-used dataset for image captioning, containing **8,000 images and 40,000 captions** (five captions per image).   
- **Dataset Preprocessing:** Captions were tokenized, padded, and transformed into sequences. Images were resized and passed through the pre-trained models for feature extraction.  

### **Model Architecture:**  
- **CNN Encoder:** Pre-trained models were used to extract image features. Models tested include:  
  - **VGG16**  
  - **ResNet50**  
  - **InceptionV3**  
  - **DenseNet201**  
  - **EfficientNetV2-B2**  
  - **ConvNeXtBase**  
  - **Custom Model (from scratch)**  
  
- **BiLSTM Decoder:** A bidirectional LSTM with 128 or 256 units was used to process sequential caption data.  
- **Attention Mechanism:** Additive attention was integrated into the decoder to enhance feature importance by aligning attention over specific regions of the image during caption generation.  

### **Parameters and Hyperparameters:**  
- **Embedding Dimension:** 256 (Increased to 512 for attention-based models)  
- **BiLSTM Units:** 128 (256 for attention-based models)  
- **Dropout Rate:** 0.5  
- **Regularization:** L2 (0.01)  
- **Batch Size:** 64  
- **Optimizer:** Adam (Learning Rate: 0.001)  
- **Epochs:** 10-15  

## 🔍 **Training and Results**  
### **Training Process:**  
- Models were trained for **10 to 15 epochs**, with early stopping based on validation loss. Training utilized the **categorical cross-entropy loss function**.
- Each model was saved at the epoch with the best performance (lowest validation loss).   

### **Performance Metrics:**  
- **BLEU (1 to 4)** – Measures precision at n-gram levels.  
- **METEOR** – Considers synonym matching and recall.  
- **ROUGE-L** – Longest common subsequence-based recall metric.  
- **CIDEr** – Consensus-based metric that compares generated captions with human references.  

### **Results:**  
| Model              | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr  |
|-------------------|--------|--------|--------|--------|--------|---------|--------|
| VGG16             | 0.4130 | 0.2007 | 0.1040 | 0.0360 | 0.1047 | 0.1547  | 0.1581 |
| ResNet50          | 0.3897 | 0.1735 | 0.0766 | 0.0286 | 0.0883 | 0.1404  | 0.0958 |
| InceptionV3       | 0.4909 | 0.3020 | 0.1813 | 0.1001 | 0.1421 | 0.2161  | 0.3596 |
| DenseNet201       | 0.4976 | 0.3120 | 0.1832 | 0.1016 | 0.1442 | 0.2140  | 0.3552 |
| DenseNet + Attn   | 0.4760 | 0.2941 | 0.1909 | 0.1145 | 0.1408 | 0.2020  | 0.3412 |
| DenseNet + Attn (512D) | 0.4856 | 0.3126 | 0.2159 | 0.1649 | 0.1990 | 0.2657  | 0.8747 |
| EfficientNetV2-B2 | 0.3740 | 0.1944 | 0.0746 | 0.0284 | 0.0850 | 0.1342  | 0.0879 |
| ConvNeXtBase      | 0.4314 | 0.2504 | 0.1516 | 0.0838 | 0.1140 | 0.1794  | 0.2256 |
| Custom Model      | 0.4207 | 0.1783 | 0.0835 | 0.0365 | 0.0848 | 0.1400  | 0.0676 |

### **Key Findings:**  
- **DenseNet201 with enhanced attention (512D embedding)** achieved the best performance, with the highest **BLEU-4 (0.1649)** and **CIDEr (0.8747)** scores.  
- **InceptionV3** and **ConvNeXtBase** showed promising results but lagged slightly in longer caption generation.  
- Attention-based models consistently reduced overfitting and improved overall caption fluency by focusing on critical image regions.  

## 🚀 **How to Use**  
### **Dependencies:**  
- Python 3.8+  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- NLTK  

### **Steps to Run:**  
1. Clone the repository.  
   ```bash
   git clone https://github.com/username/image-captioning
   cd image-captioning
   ```  
2. Install dependencies.  
   ```bash
   pip install -r requirements.txt
   ```  
3. Download and preprocess the Flickr 8k dataset.  
4. Run the training script.  
   ```bash
   python train.py
   ```  
5. Evaluate model performance.  
   ```bash
   python evaluate.py
   ```  

## 📄 **Conclusion:**  
This project demonstrates the effectiveness of combining **CNNs, BiLSTM decoders, and attention mechanisms** for image captioning. The results emphasize the importance of deeper architectures and attention mechanisms for generating accurate, fluent, and human-like captions.

