# Phase Retrieval Deep Learning

In the field of computational imaging (CI), supervised training methods have long been the dominant approach for neural networks in optics. These methods heavily rely on large amounts of labeled data to adjust network weights and biases effectively. However, obtaining a substantial number of ground-truth images for training poses significant challenges in real-world scenarios like phase retrieval or 2D/3D imaging applications. To overcome this limitation, we propose an innovative approach that merges principles from physics with deep neural networks. Our objective is to reduce the dependency on extensive labeled data by incorporating a comprehensive physical model that accurately represents the image formation process. This unique approach allows us to achieve 3D imaging through phase retrieval, utilizing techniques such as Gerchberg-Saxton (GS) and Fourier-Rytov (FR), in combination with deep learning architectures to extract intricate information from the phase. Consequently, this information enables us to detect changes in an object's surface and generate a mesh representation of its 2D/3D structure. In our proposal,  we introduce Res-U2Net, a novel untrained neural network designed to estimate the 3D structure of objects. By adopting a unified method for object analysis, this approach presents a new paradigm for neural network design, seamlessly integrating physical models. Furthermore, this framework can be extended to address a wide range of other computational imaging challenges.

@article{OsorioQuero:24,
author = {Carlos Osorio Quero and Daniel Leykam and Irving Rondon Ojeda},
journal = {J. Opt. Soc. Am. A},
keywords = {Biomedical imaging; Computational imaging; Fluorescence lifetime imaging; Imaging techniques; Inverse design; Phase retrieval},
number = {5},
pages = {766--773},
publisher = {Optica Publishing Group},
title = {Res-U2Net: untrained deep learning for phase retrieval and image reconstruction},
volume = {41},
month = {May},
year = {2024},
url = {https://opg.optica.org/josaa/abstract.cfm?URI=josaa-41-5-766},
doi = {10.1364/JOSAA.511074}
}


