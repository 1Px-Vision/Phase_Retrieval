In the field of computational imaging (CI), supervised training methods have long been the dominant approach for neural networks in optics. These methods heavily rely on large amounts of 
labeled data to adjust network weights and biases effectively. However, obtaining a substantial number of ground-truth images for training poses significant challenges in real-world scenarios 
like phase retrieval or 2D/3D imaging applications. To overcome this limitation, we propose an innovative approach that merges principles from physics with deep neural networks. Our objective 
is to reduce the dependency on extensive labeled data by incorporating a comprehensive physical model that accurately represents the image formation process. This unique approach allows us to 
achieve 3D imaging through phase retrieval, utilizing techniques such as Gerchberg-Saxton (GS) and Fourier-Rytov (FR), in combination with deep learning architectures to extract intricate 
information from the phase. Consequently, this information enables us to detect changes in an object's surface and generate a mesh representation of its 2D/3D structure. In our proposal, 
we introduce Res-U2Net, a novel untrained neural network designed to estimate the 3D structure of objects. By adopting a unified method for object analysis, this approach presents a 
new paradigm for neural network design, seamlessly integrating physical models. Furthermore, this framework can be extended to address a wide range of other computational imaging challenges
