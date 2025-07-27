{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# Install necessary libraries\n",
        "!pip install transformers datasets pandas scikit-learn torch accelerate -q\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n6o9XognCtOR",
        "outputId": "927e4a8c-2139-474c-8d38-6ad796c432fc"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[33mWARNING: Ignoring invalid distribution ~vidia-cublas-cu12 (/usr/local/lib/python3.11/dist-packages)\u001b[0m\u001b[33m\n",
            "\u001b[0m\u001b[33mWARNING: Ignoring invalid distribution ~vidia-cublas-cu12 (/usr/local/lib/python3.11/dist-packages)\u001b[0m\u001b[33m\n",
            "\u001b[0m\u001b[33mWARNING: Ignoring invalid distribution ~vidia-cublas-cu12 (/usr/local/lib/python3.11/dist-packages)\u001b[0m\u001b[33m\n",
            "\u001b[0m\u001b[33mWARNING: Ignoring invalid distribution ~vidia-cublas-cu12 (/usr/local/lib/python3.11/dist-packages)\u001b[0m\u001b[33m\n",
            "\u001b[0m"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, GPT2LMHeadModel, GPT2Tokenizer\n",
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from tqdm.notebook import tqdm\n",
        "import os"
      ],
      "metadata": {
        "id": "sX3NCs3qFSl7"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "try:\n",
        "    df_fake = pd.read_csv('fake.csv')\n",
        "    df_true = pd.read_csv('true.csv')\n",
        "\n",
        "    # Add a 'label' column to each DataFrame\n",
        "    df_fake['label'] = 'FAKE'\n",
        "    df_true['label'] = 'REAL'\n",
        "\n",
        "    # Combine the datasets\n",
        "    df = pd.concat([df_fake, df_true], ignore_index=True)\n",
        "    print(\"Datasets loaded and combined successfully!\")\n",
        "    print(df.head())\n",
        "    print(f\"\\nDataset shape: {df.shape}\")\n",
        "    print(f\"Labels distribution:\\n{df['label'].value_counts()}\")\n",
        "\n",
        "except FileNotFoundError:\n",
        "    print(\"Error: 'fake.csv' or 'true.csv' not found. Please ensure the datasets are uploaded to 'data/' or downloaded.\")\n",
        "    # Create a dummy DataFrame for demonstration if the files are not found\n",
        "    df = pd.DataFrame({\n",
        "        'id': range(5),\n",
        "        'title': [\n",
        "            \"Donald Trump Jr. Calls For 'Total Transparency' In Russia Probe\",\n",
        "            \"Hillary Clinton's Email Scandal Deepens\",\n",
        "            \"Scientists Discover New Planet With Life\",\n",
        "            \"BREAKING: Aliens Land In Washington D.C.\",\n",
        "            \"New Study Shows Coffee Prevents All Diseases\"\n",
        "        ],\n",
        "        'text': [\n",
        "            \"Donald Trump Jr. on Sunday called for 'total transparency' in the ongoing investigation into Russian interference in the 2016 election...\",\n",
        "            \"The controversy surrounding Hillary Clinton's use of a private email server during her tenure as Secretary of State continues to unfold...\",\n",
        "            \"Astronomers at the Kepler Space Telescope have announced the discovery of a new exoplanet, Kepler-186f, which appears to be in the habitable zone...\",\n",
        "            \"Reports are flooding in from Washington D.C. of an unprecedented event: multiple unidentified flying objects have landed near the Lincoln Memorial...\",\n",
        "            \"A groundbreaking new study published in the Journal of Health claims that daily consumption of coffee can prevent all known human diseases, from cancer to the common cold...\"\n",
        "        ],\n",
        "        'label': ['REAL', 'REAL', 'REAL', 'FAKE', 'FAKE']\n",
        "    })\n",
        "    print(\"\\nUsing dummy data for demonstration.\")\n",
        "\n",
        "\n",
        "# Preprocessing: Map labels to numerical values\n",
        "# 'REAL' -> 0, 'FAKE' -> 1\n",
        "df['label_encoded'] = df['label'].apply(lambda x: 0 if x == 'REAL' else 1)\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "# We'll use the 'text' column for classification, as it's more substantial than just the title.\n",
        "train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])\n",
        "\n",
        "print(f\"\\nTrain set shape: {train_df.shape}\")\n",
        "print(f\"Test set shape: {test_df.shape}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f9qIrLUXBH0l",
        "outputId": "e117ae60-4702-428c-9f39-d8aaae4100c8"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Datasets loaded and combined successfully!\n",
            "                                               title  \\\n",
            "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
            "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
            "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
            "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
            "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
            "\n",
            "                                                text subject  \\\n",
            "0  Donald Trump just couldn t wish all Americans ...    News   \n",
            "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
            "2  On Friday, it was revealed that former Milwauk...    News   \n",
            "3  On Christmas day, Donald Trump announced that ...    News   \n",
            "4  Pope Francis used his annual Christmas Day mes...    News   \n",
            "\n",
            "                date label  \n",
            "0  December 31, 2017  FAKE  \n",
            "1  December 31, 2017  FAKE  \n",
            "2  December 30, 2017  FAKE  \n",
            "3  December 29, 2017  FAKE  \n",
            "4  December 25, 2017  FAKE  \n",
            "\n",
            "Dataset shape: (44898, 5)\n",
            "Labels distribution:\n",
            "label\n",
            "FAKE    23481\n",
            "REAL    21417\n",
            "Name: count, dtype: int64\n",
            "\n",
            "Train set shape: (35918, 6)\n",
            "Test set shape: (8980, 6)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Load pre-trained GPT-2 model and tokenizer\n",
        "print(\"Loading GPT-2 model and tokenizer...\")\n",
        "gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')\n",
        "gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')\n",
        "\n",
        "# Add the padding token for GPT-2 if it's not set, which is common for generation\n",
        "if gpt2_tokenizer.pad_token is None:\n",
        "    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token\n",
        "\n",
        "print(\"GPT-2 model loaded.\")\n",
        "\n",
        "def generate_fake_headline(prompt, max_length=50, num_return_sequences=1, temperature=0.9, top_k=50, top_p=0.95):\n",
        "    \"\"\"\n",
        "    Generates a fake news headline using GPT-2 based on a given prompt.\n",
        "\n",
        "    Args:\n",
        "        prompt (str): The starting text for the headline.\n",
        "        max_length (int): Maximum length of the generated sequence.\n",
        "        num_return_sequences (int): Number of headlines to generate.\n",
        "        temperature (float): Controls randomness. Lower values make output more deterministic.\n",
        "        top_k (int): Top-K sampling. Only consider the top K most likely next tokens.\n",
        "        top_p (float): Top-P (nucleus) sampling. Only consider tokens whose cumulative probability exceeds p.\n",
        "\n",
        "    Returns:\n",
        "        list: A list of generated headlines.\n",
        "    \"\"\"\n",
        "    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')\n",
        "\n",
        "    # Generate text\n",
        "    output = gpt2_model.generate(\n",
        "        input_ids,\n",
        "        max_length=max_length,\n",
        "        num_return_sequences=num_return_sequences,\n",
        "        temperature=temperature,\n",
        "        do_sample=True, # Enable sampling for more diverse outputs\n",
        "        top_k=top_k,\n",
        "        top_p=top_p,\n",
        "        pad_token_id=gpt2_tokenizer.eos_token_id # Use EOS token for padding\n",
        "    )\n",
        "\n",
        "    generated_headlines = []\n",
        "    for i, sample_output in enumerate(output):\n",
        "        decoded_text = gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)\n",
        "        # Remove the prompt from the generated text\n",
        "        headline = decoded_text[len(prompt):].strip()\n",
        "        # Take only the first sentence or a reasonable portion for a headline\n",
        "        headline = headline.split('\\n')[0].split('.')[0].strip()\n",
        "        generated_headlines.append(headline)\n",
        "    return generated_headlines\n",
        "\n",
        "# --- Demonstration of Headline Generation ---\n",
        "print(\"\\n--- Generating Fake Headlines ---\")\n",
        "prompt1 = \"BREAKING NEWS: Scientists discover\"\n",
        "headlines1 = generate_fake_headline(prompt1, num_return_sequences=3, max_length=30)\n",
        "print(f\"Prompt: '{prompt1}'\")\n",
        "for i, h in enumerate(headlines1):\n",
        "    print(f\"  Generated Headline {i+1}: {h}\")\n",
        "\n",
        "print(\"-\" * 30)\n",
        "\n",
        "prompt2 = \"Urgent: New report reveals\"\n",
        "headlines2 = generate_fake_headline(prompt2, num_return_sequences=2, max_length=40, temperature=1.0)\n",
        "print(f\"Prompt: '{prompt2}'\")\n",
        "for i, h in enumerate(headlines2):\n",
        "    print(f\"  Generated Headline {i+1}: {h}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HAVVUskrBxTl",
        "outputId": "9667ae43-2ebe-4a9f-c56a-f99cafd32dc8"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading GPT-2 model and tokenizer...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n",
            "The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "GPT-2 model loaded.\n",
            "\n",
            "--- Generating Fake Headlines ---\n",
            "Prompt: 'BREAKING NEWS: Scientists discover'\n",
            "  Generated Headline 1: a 'massive crater' about 100km from the Moon, the scientists said Tuesday\n",
            "  Generated Headline 2: a genetic difference in how the brains of elephants and elephants react to noise\n",
            "  Generated Headline 3: new mechanism in dinosaur's teeth that may explain why they look like normal pups when their bodies become damaged\n",
            "------------------------------\n",
            "Prompt: 'Urgent: New report reveals'\n",
            "  Generated Headline 1: a spike in shootings after Newtown\n",
            "  Generated Headline 2: 'significant financial, material and operational costs' to the Pentagon from the war against ISIL\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define a custom Dataset class for PyTorch\n",
        "class NewsDataset(Dataset):\n",
        "    def __init__(self, texts, labels, tokenizer, max_len):\n",
        "        self.texts = texts\n",
        "        self.labels = labels\n",
        "        self.tokenizer = tokenizer\n",
        "        self.max_len = max_len\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.texts)\n",
        "\n",
        "    def __getitem__(self, item):\n",
        "        text = str(self.texts[item])\n",
        "        label = self.labels[item]\n",
        "\n",
        "        encoding = self.tokenizer.encode_plus(\n",
        "            text,\n",
        "            add_special_tokens=True,\n",
        "            max_length=self.max_len,\n",
        "            return_token_type_ids=False,\n",
        "            padding='max_length',\n",
        "            truncation=True,\n",
        "            return_attention_mask=True,\n",
        "            return_tensors='pt',\n",
        "        )\n",
        "\n",
        "        return {\n",
        "            'text': text,\n",
        "            'input_ids': encoding['input_ids'].flatten(),\n",
        "            'attention_mask': encoding['attention_mask'].flatten(),\n",
        "            'labels': torch.tensor(label, dtype=torch.long)\n",
        "        }\n",
        "\n",
        "# Load pre-trained BERT tokenizer and model for sequence classification\n",
        "print(\"Loading BERT tokenizer and model for sequence classification...\")\n",
        "bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n",
        "# We have 2 labels: REAL (0) and FAKE (1)\n",
        "bert_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)\n",
        "\n",
        "# Move model to GPU if available\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "bert_model.to(device)\n",
        "print(f\"BERT model loaded and moved to: {device}\")\n",
        "\n",
        "# --- Prepare data for BERT inference (or potential fine-tuning) ---\n",
        "MAX_LEN = 512 # Max sequence length for BERT\n",
        "\n",
        "# Create datasets\n",
        "train_dataset = NewsDataset(\n",
        "    texts=train_df['text'].to_numpy(),\n",
        "    labels=train_df['label_encoded'].to_numpy(),\n",
        "    tokenizer=bert_tokenizer,\n",
        "    max_len=MAX_LEN\n",
        ")\n",
        "\n",
        "test_dataset = NewsDataset(\n",
        "    texts=test_df['text'].to_numpy(),\n",
        "    labels=test_df['label_encoded'].to_numpy(),\n",
        "    tokenizer=bert_tokenizer,\n",
        "    max_len=MAX_LEN\n",
        ")\n",
        "\n",
        "# Create data loaders\n",
        "BATCH_SIZE = 8 # Smaller batch size for demonstration\n",
        "train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
        "test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
        "\n",
        "print(f\"\\nData loaded for BERT (Train: {len(train_dataset)}, Test: {len(test_dataset)})\")\n",
        "\n",
        "# --- Function to predict if a text is fake or real ---\n",
        "def predict_fake_news(text_to_classify, model, tokenizer, device, max_len=MAX_LEN):\n",
        "    \"\"\"\n",
        "    Predicts whether a given text is fake or real using the BERT model.\n",
        "\n",
        "    Args:\n",
        "        text_to_classify (str): The text to classify.\n",
        "        model: The pre-trained BERT classification model.\n",
        "        tokenizer: The BERT tokenizer.\n",
        "        device: The device (cpu or cuda) to run the model on.\n",
        "        max_len (int): Maximum sequence length for tokenization.\n",
        "\n",
        "    Returns:\n",
        "        str: 'FAKE' or 'REAL'.\n",
        "        float: Probability of being 'FAKE'.\n",
        "    \"\"\"\n",
        "    model.eval() # Set model to evaluation mode\n",
        "\n",
        "    encoding = tokenizer.encode_plus(\n",
        "        text_to_classify,\n",
        "        add_special_tokens=True,\n",
        "        max_length=max_len,\n",
        "        return_token_type_ids=False,\n",
        "        padding='max_length',\n",
        "        truncation=True,\n",
        "        return_attention_mask=True,\n",
        "        return_tensors='pt',\n",
        "    )\n",
        "\n",
        "    input_ids = encoding['input_ids'].to(device)\n",
        "    attention_mask = encoding['attention_mask'].to(device) # Corrected typo here\n",
        "\n",
        "    with torch.no_grad():\n",
        "        outputs = model(input_ids=input_ids, attention_mask=attention_mask)\n",
        "        logits = outputs.logits\n",
        "        probabilities = torch.softmax(logits, dim=1) # Convert logits to probabilities\n",
        "\n",
        "    # Get the predicted class (0 for REAL, 1 for FAKE)\n",
        "    predicted_class_id = torch.argmax(probabilities, dim=1).item()\n",
        "    fake_probability = probabilities[0][1].item() # Probability of the 'FAKE' class (index 1)\n",
        "\n",
        "    prediction_label = 'FAKE' if predicted_class_id == 1 else 'REAL'\n",
        "    return prediction_label, fake_probability\n",
        "\n",
        "# --- Demonstration of BERT Prediction ---\n",
        "print(\"\\n--- Demonstrating BERT Prediction on Sample Texts ---\")\n",
        "\n",
        "# Example 1: A real news text from the dataset\n",
        "sample_real_text = test_df[test_df['label'] == 'REAL']['text'].iloc[0]\n",
        "print(f\"\\nText Sample (REAL):\\n{sample_real_text[:150]}...\")\n",
        "prediction, prob = predict_fake_news(sample_real_text, bert_model, bert_tokenizer, device)\n",
        "print(f\"Prediction: {prediction} (Probability of FAKE: {prob:.4f})\")\n",
        "\n",
        "# Example 2: A fake news text from the dataset\n",
        "sample_fake_text = test_df[test_df['label'] == 'FAKE']['text'].iloc[0]\n",
        "print(f\"\\nText Sample (FAKE):\\n{sample_fake_text[:150]}...\")\n",
        "prediction, prob = predict_fake_news(sample_fake_text, bert_model, bert_tokenizer, device)\n",
        "print(f\"Prediction: {prediction} (Probability of FAKE: {prob:.4f})\")\n",
        "\n",
        "# Note: Without fine-tuning, the pre-trained BERT model might not be highly accurate\n",
        "# on this specific fake news detection task, as it hasn't learned the nuances of\n",
        "# this dataset's fake vs. real patterns. Fine-tuning would significantly improve performance."
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rTqys0YpB39R",
        "outputId": "6b8ea1ae-f0f9-452a-c382-a73825f5b131"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading BERT tokenizer and model for sequence classification...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "BERT model loaded and moved to: cpu\n",
            "\n",
            "Data loaded for BERT (Train: 35918, Test: 8980)\n",
            "\n",
            "--- Demonstrating BERT Prediction on Sample Texts ---\n",
            "\n",
            "Text Sample (REAL):\n",
            "WASHINGTON (Reuters) - U.S. President Donald Trump will nominate Goldman Sachs (GS.N) banker James Donovan as deputy Treasury secretary, the White Hou...\n",
            "Prediction: FAKE (Probability of FAKE: 0.6007)\n",
            "\n",
            "Text Sample (FAKE):\n",
            "Amateur president Donald Trump s hostility towards the Environmental Protection Agency is revealing by his pick to head the EPA who attacked his own a...\n",
            "Prediction: FAKE (Probability of FAKE: 0.6287)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"\\n--- Integrating GPT-2 Generation and BERT Detection ---\")\n",
        "\n",
        "# Generate a headline using GPT-2\n",
        "prompt_for_integration = \"Exclusive: Secret government documents reveal\"\n",
        "generated_headline = generate_fake_headline(prompt_for_integration, max_length=60, num_return_sequences=1, temperature=0.8)[0]\n",
        "\n",
        "full_text_to_classify = f\"{generated_headline}. This astonishing revelation comes after years of speculation...\"\n",
        "\n",
        "print(f\"\\nGPT-2 Generated Headline (based on prompt '{prompt_for_integration}'):\")\n",
        "print(f\"  '{generated_headline}'\")\n",
        "\n",
        "# Classify the generated headline using BERT\n",
        "print(\"\\nClassifying the generated text with BERT:\")\n",
        "prediction_label, fake_probability = predict_fake_news(full_text_to_classify, bert_model, bert_tokenizer, device)\n",
        "\n",
        "print(f\"  Classification Result: {prediction_label}\")\n",
        "print(f\"  Probability of being FAKE: {fake_probability:.4f}\")\n",
        "\n",
        "print(\"\\n--- Ethical Considerations ---\")\n",
        "print(\"This project demonstrates the dual nature of advanced AI models:\")\n",
        "print(\"1.  **Generative Power (GPT-2):** Capable of creating highly realistic text, which can be misused to spread misinformation.\")\n",
        "print(\"2.  **Detection Power (BERT):** Essential for building tools to identify and combat such misinformation.\")\n",
        "print(\"\\nIt underscores the ethical responsibility in developing and deploying AI, emphasizing the need for robust detection mechanisms alongside powerful generation capabilities to maintain a trustworthy information ecosystem.\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7XQXTv6KB8s5",
        "outputId": "ad4a32e8-edfd-4290-fd00-0c916cca38ba"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "--- Integrating GPT-2 Generation and BERT Detection ---\n",
            "\n",
            "GPT-2 Generated Headline (based on prompt 'Exclusive: Secret government documents reveal'):\n",
            "  'how the U'\n",
            "\n",
            "Classifying the generated text with BERT:\n",
            "  Classification Result: FAKE\n",
            "  Probability of being FAKE: 0.5997\n",
            "\n",
            "--- Ethical Considerations ---\n",
            "This project demonstrates the dual nature of advanced AI models:\n",
            "1.  **Generative Power (GPT-2):** Capable of creating highly realistic text, which can be misused to spread misinformation.\n",
            "2.  **Detection Power (BERT):** Essential for building tools to identify and combat such misinformation.\n",
            "\n",
            "It underscores the ethical responsibility in developing and deploying AI, emphasizing the need for robust detection mechanisms alongside powerful generation capabilities to maintain a trustworthy information ecosystem.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "vGSRS1LFD6Ma"
      },
      "execution_count": 6,
      "outputs": []
    }
  ]
}