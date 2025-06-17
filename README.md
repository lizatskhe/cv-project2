# Project 2: Improving Robustness of Deepfake Detectors through Gradient Regularization

This project explores improving the robustness of deepfake detectors using **gradient regularization**, a technique that enhances model generalization by incorporating shallow feature statistics during training. The baseline model is an EfficientNet-B0 trained on a combined dataset of **DFFD and FaceForensics++**, and evaluated against adversarial attacks such as **FGSM** and **PGD**.

## 📌 Project Goals

- Train a deepfake detection model using EfficientNet-B0.
- Evaluate its accuracy and generalization ability.
- Test model robustness under adversarial attacks (FGSM, PGD).
- Implement and assess **gradient regularization**.

## 📦 Dataset
Due to the class imbalance present in the DFFD dataset, we instead extracted video frames from the FaceForensics++ dataset and created a more balanced dataset by combining real and fake samples.

The dataset can be downloaded from the following link: 

https://drive.google.com/file/d/16mWUoUlmtfPbjMl7lCmlURSH-qmGe1uN/view?usp=sharing.

## 📁 Project Structure

```project/
  CV2_Alhalawani_Tskhe.ipynb  
  CV2_Alhalawani_Tskhe.pptx
  baseline/
    baseline_model_5_combined.pth       # baseline model
    fgsm/                               # FGSM attacks on baseline model
      fgsm_eps_0.01_combined.pt
      fgsm_eps_0.02_combined.pt
      fgsm_eps_0.001_combined.pt
      fgsm_eps_0.005_combined.pt
    pgd/                                # PGD attacks on baseline model
      pgd_eps_0.001_steps_5_combined.pt
      pgd_eps_0.0001_steps_5_combined.pt
      pgd_eps_0.0005_steps_5_combined.pt
  grad_reg/
    model/                              # models with gradient regularization
      model_epoch_1.pth
      model_epoch_2.pth
      model_epoch_4.pth
      model_epoch_6.pth                 # model used for evaluation
      model_epoch_8.pth
      model_epoch_10.pth
      final-path.pth
    fgsm/                               # FGSM attacks on model with gradient regularization
      fgsm_results_eps_0.0100.pt
      fgsm_results_eps_0.0200.pt
      fgsm_results_eps_0.0010.pt
      fgsm_results_eps_0.0050.pt
    pgd/                                # PGD attacks on model with gradient regularization
      pgd_results_eps_0.0010.pt
      pgd_results_eps_0.0001.pt
      pgd_results_eps_0.0005.pt
```

