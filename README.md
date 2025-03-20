# **AFAGC-GinNet: Lightweight Adaptive Deep Learning for Real-Time Speech Enhancement**  

🚀 **This repository contains the implementation of our work (under review), presenting a lightweight deep learning model for real-time speech enhancement on edge devices.**  

---

## **📌 Status: Under Review**  
📄 **Title:** *Lightweight Adaptive Deep Learning for Efficient Real-Time Speech Enhancement on Edge Devices*  
✏️ **Authors:** Fazal E Wahab, Zhongfu Ye, Nasir Saleem, Sami Bourious, and Amir Hussain  
📚 **Status:** *Preprint / Under Review*  
🔗 **Paper Link:** *(Coming soon)*  

---

## **📖 Dataset**  
This project used the VoiceBank-DEMAND dataset. This dataset consists of 30 speakers from the VoiceBank corpus, which is further divided into a training set (28 speakers) and a testing set (2 speakers).

Training Set: 11,572 utterances from 28 speakers mixed with DEMAND noises and artificial background noises at SNRs of 0, 5, 10, and 15 dB.
Testing Set: 824 utterances from 2 unseen speakers mixed with unseen DEMAND noises at SNRs of 2.5, 7.5, 12.5, and 17.5 dB.
📌 please download the dataset from VoiceBank-DEMAND.
```bash
Dataset Folder Structure
Once the dataset is prepared, it should follow this folder structure:
datasets
│── cv
│   └── cv.ex
│── tr
│   ├── tr_0.ex
│   ├── tr_1.ex
│   ├── tr_2.ex
│   ├── tr_3.ex
│   └── tr_4.ex
│── tt
│   ├── tt_snr0.ex
│   ├── tt_snr-5.ex
│   └── tt_snr5.ex

---

---

## **🚀 Getting Started**  

### **1️⃣ Installation**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/your-username/AFAGC-GinNet.git
cd AFAGC-GinNet
pip install -r requirements.txt

---

## **2️⃣ Dataset
This project used the VoiceBank-DEMAND dataset. This dataset consists of 30 speakers from the VoiceBank corpus, which is further divided into a training set (28 speakers) and a testing set (2 speakers).

Training Set: 11,572 utterances from 28 speakers mixed with DEMAND noises and artificial background noises at SNRs of 0, 5, 10, and 15 dB.
Testing Set: 824 utterances from 2 unseen speakers mixed with unseen DEMAND noises at SNRs of 2.5, 7.5, 12.5, and 17.5 dB.
📌 please download the dataset from VoiceBank-DEMAND.

---

Dataset Folder Structure
Once the dataset is prepared, it should follow this folder structure:
datasets
│── cv
│   └── cv.ex
│── tr
│   ├── tr_0.ex
│   ├── tr_1.ex
│   ├── tr_2.ex
│   ├── tr_3.ex
│   └── tr_4.ex
│── tt
│   ├── tt_snr0.ex
│   ├── tt_snr-5.ex
│   └── tt_snr5.ex

