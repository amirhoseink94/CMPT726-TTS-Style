# Stylistic Text To Speech Synthesize

This is the final project of the course "CMPT726 G100 Machine Learning" at Simon Fraser university.

In this project we propose the application of two generative networks for learning emotional style transfer to be used in text-to-speech. Our approach borrows from a generative flow-based model (Glow) and a voice conversion method (MelGANVC) inspired by a generative adversarial network shown to generate high quality audio. Glow is trained by maximizing the likelihood of the training data, and MelGAN-VC is trained by minimizing both an adversarial loss and a Siamese
margin-based contrastive loss. Our mean opinion scores (MOS) show that Glow was more effective at emotional style transfer compared to MelGAN-VC. Our
results are promising for improved data-efficiency as Glow was trained on 0.8% of the data used to train MelGAN-VC.


