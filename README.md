# ML Challenge Problem Statement

## Feature Extraction from Images

In this hackathon, the goal is to create a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital. As digital marketplaces expand, many products lack detailed textual descriptions, making it essential to obtain key details directly from images. These images provide important information such as weight, volume, voltage, wattage, dimensions, and many more, which are critical for digital stores.

### Data Description: 

The dataset consists of the following columns: 

1. **index:** An unique identifier (ID) for the data sample
2. **image_link**: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
3. **group_id**: Category code of the product
4. **entity_name:** Product entity name. For eg: “item_weight” 
5. **entity_value:** Product entity value. For eg: “34 gram” 

# Image Text Extraction and Entity Recognition

## **Step 1: Image Text Extraction using PaddleOCR**

### **Tools**:
- PaddleOCR
- OpenCV
- Python Requests

The project begins by downloading and processing images using OpenCV. Text is extracted from these images using PaddleOCR, which facilitates the detection and recognition of text in English.

- **Function**: `extract_text_from_image()`  
  - Downloads the image using Python Requests.
  - Applies PaddleOCR to capture the textual content from the image.

---

## **Step 2: Text Embeddings using BERT**

### **Tools**:
- Hugging Face Transformers (BERT)
- PyTorch

The next step involves transforming the extracted text into embeddings using the pre-trained **`bert-base-uncased`** model.

- **Process**:
  1. Tokenize the extracted text.
  2. Pass the tokenized text through the BERT model.
  3. Generate contextual embeddings (vector representations) for each sentence.  
These embeddings are later used for entity classification.

---

## **Step 3: Data Preparation and BART Training**

### **Tools**:
- Hugging Face Transformers (BART)
- PyTorch
- Sklearn

For entity recognition, the extracted text embeddings are passed through a **BART-based model** fine-tuned for sequence classification.

- **Process**:
  1. Split the dataset into training and test sets using `train_test_split()`.
  2. Tokenize input sentences and target labels.
  3. Train the BART model over multiple epochs using the AdamW optimizer.  
     - Padded sequences are used to handle variable-length inputs.
     - Training loops iterate over the dataset in batches for fine-tuning.

---

## **Step 4: Evaluation**

### **Metrics**:
- **BLEU Score**: Measures the quality of predictions compared to actual target sequences.
- **Exact Match Accuracy**: Checks for exact matches between predicted and actual sequences.

- **Results**:
  - **Exact Match Accuracy**: 64.78%  
    Indicates the percentage of correct predictions made by the BART model.
  - **Average BLEU Score**: 0.2549  
    Reflects how closely the predicted sequences resemble the actual values.

---

## **Results and Outcomes**

### **Model Performance**:
- **Exact Match Accuracy**: 64.78%
- **Average BLEU Score**: 0.2549

The project successfully integrates image text extraction, text embeddings, and fine-tuned sequence classification to achieve effective entity recognition.

