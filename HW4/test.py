{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNX2GlbH1I9eyXmve0ctecm",
      "include_colab_link": true
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
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/laynotena/Data-Mining/blob/main/HW4/test.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SZDAVpnUA0o6"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from keras.preprocessing import sequence\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "import tensorflow as tf"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YPeqVJagB-c_"
      },
      "source": [
        "training = pd.read_csv('training_label.txt', delimiter = \"\\t\",names = ['emoticon'])\n",
        "testing = pd.read_csv('testing_label.txt',delimiter = \"\\t\", names = ['emoticon'])"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "HW0Bk5CHCaEm",
        "outputId": "f9230b75-50e2-4b92-bbbb-518bb91e5e11"
      },
      "source": [
        "def data_generate(data):\n",
        "    label = []\n",
        "    text = []\n",
        "    for i in range(len(data)):\n",
        "        label.append(data.emoticon[i][0])\n",
        "        text.append(data.emoticon[i][9:])\n",
        "    df = pd.DataFrame(data = label, columns=['label'])\n",
        "    df['text'] = text\n",
        "    return df\n",
        "\n",
        "train = data_generate(training)\n",
        "train.head()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>label</th>\n",
              "      <th>text</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>are wtf ... awww thanks !</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>leavingg to wait for kaysie to arrive myspaci...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0</td>\n",
              "      <td>i wish i could go and see duffy when she come...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>i know eep ! i can ' t wait for one more day ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>so scared and feeling sick . fuck ! hope some...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  label                                               text\n",
              "0     1                          are wtf ... awww thanks !\n",
              "1     1   leavingg to wait for kaysie to arrive myspaci...\n",
              "2     0   i wish i could go and see duffy when she come...\n",
              "3     1   i know eep ! i can ' t wait for one more day ...\n",
              "4     0   so scared and feeling sick . fuck ! hope some..."
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KIOABynsGLqL"
      },
      "source": [
        "# 新增區段"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CVPJZ_XPF8sI"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Hpb4hWw-Coa2"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}