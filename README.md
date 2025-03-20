## **📖 Dataset**  
This project used the [**VoiceBank-DEMAND**](https://datashare.ed.ac.uk/handle/10283/2791) dataset. This dataset consists of **30 speakers** from the VoiceBank corpus, which is further divided into a **training set (28 speakers)** and a **testing set (2 speakers)**.  

- **Training Set:** 11,572 utterances from 28 speakers mixed with **DEMAND noises** and artificial background noises at SNRs of **0, 5, 10, and 15 dB**.  
- **Testing Set:** 824 utterances from 2 unseen speakers mixed with **unseen DEMAND noises** at SNRs of **2.5, 7.5, 12.5, and 17.5 dB**.  

📌 To **train/test**, please download the dataset from [**VoiceBank-DEMAND**](https://datashare.ed.ac.uk/handle/10283/2791).  

### **Dataset Folder Structure**  
Once the dataset is prepared, it should follow this folder structure:  

```bash
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

### 🔥 **Why This Fix Works?**  
1️⃣ **A blank line** after the code block helps Markdown recognize it as a complete section.  
2️⃣ **A horizontal rule (`---`)** **forces GitHub Markdown to visually separate the sections.**  
3️⃣ **Ensures "Getting Started" starts on a new line** without merging into the dataset folder structure.

---

### **🎯 Final Answer: What Should You Do?**
- **Copy & paste** the **updated Markdown** above into your `README.md`.  
- **GitHub will now correctly separate the sections** when you view your repository.  

🚀 **Now it will display perfectly!** 😊🎯
