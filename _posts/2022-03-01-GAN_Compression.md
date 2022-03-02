---
title: GAN Compression(Efficient Architectures for Interactive Conditional GANs)
tags:
    - Deep Learning
    - Vision
    - Generative
    - Paper
---

GAN Compression: Efficient Architectures for Interactive Conditional GANs는 2020년 CVPR에 게재된 논문으로, CGAN의 compression을 위한 method를 제시한 논문입니다.<br> CGAN은 mobileNet과 같은 image recognition과 비교해서 계산량이 큽니다. 따라서 CGAN에 대해 inference time과 model size를 줄이기 위한 방식을 제시합니다. <br>

---

<!--more-->

#### Introduction

GAN의 경우 human interactive한 영역에서 많이 활용되지만, edge device는 hardware 성능에 한계가 있기 때문에 많은 계산량을 필요로하는 model에 대해서는 bottleneck이 생깁니다. CycleGAN과 같은 generative model들은 계산량을 많이 필요로 합니다. 따라서 GAN compression이 제안됩니다. <br>

<p>
<center><img src="/images/GAN_compression/Compression_computation_magnitude.jpg" width="400"></center>
<center><em>Fig n.</em></center>
</p>

Generative model을 compression 하는데는 2가지 근본적인 어려움이 있습니다. <br>

1. Unstability of GAN training (especially unpaired)<br> 
    Pseudo pair를 만들어 훈련시킵니다. <br>

2. Architecture가 recognition CNN과 다르다. <br> 
    중간 representation만 teacher model에서 student model로 transfer합니다. <br>

그리고 fewer cost model을 찾기 위해 NAS(Network Architecture Search)를 사용합니다. <br>

#### Related work

- Conditional GAN <br> 
    GAN(Generative Adversarial Networks)은 photo-realistic image를 합성하는데 좋은 성능을 보입니다. Conditioanl GAN은 image, label, text와 같은 다양한 conditional input을 주어 이미지 합성을 제어할 수 있도록 합니다. 고해상도의 photo-realistic image 생성은 많은 계산량을 필요로 합니다. 이로 인해 제한된 계산 리소스가 주어진 edge device에 이러한 모델을 배포하기가 어렵습니다. 따라서 interactive application을 위한 효율적인 CGAN architecture에 중점을 둡니다. <br>

- Model Acceleration <br> 
    Network model에서 필요하지 않은(중복된) 부분을 없애기 위해, network connection이나 weight에 대한 pruning을 할 수 있다. <br> 
    AMC(AutoML for Model Compression)은

- Knowledge distillation <br>

- NAS(Neural Architecture Search) <br>

#### Method

- Training Objective <br>
    CGAN은 source domain $$X$$에서 target domain $$Y$$로의 mapping $$G$$를 훈련시킵니다. CGAN의 training data는 paired와 unpaired 두가지 방식이 있기 때문에 많은model에서 paired와 unpaired를 구별하지 않고 objective function을 구성합니다.General-purpose compression에서는 teacher structure가 어떤 방식으로 training 됐는지에 관계없이 model compression이 가능하도록 paired와 unpaired를 통합했습니다. <br>

    <p>
    <center><img src="/images/GAN_compression/Compression_framework.jpg" width="600"></center>
    <center><em>Fig n.</em></center>
    </p>
    
    Origin teacher generator를 $$G'$$라고 가정합니다. <br> Unpaired data의 경우 compression과정에서 $$G'$$로 generate된 이미지를 student generator $$G$$의 pseudo GT로 사용합니다. <br>

    $$
    \begin{align}
    \mathcal{L}_{recon} =
    \begin{cases}
    \mathbb{E}_{x,y}\parallel G(x)-y\parallel, & \text{if paired CGAN} \\
    \mathbb{E}_{x}\parallel G(x)-G'(x)\parallel, & \text{if unpaired CGAN}
    \end{cases}
    \end{align}
    $$
    
    Discriminator의 경우 $$G'$$를 학습하면서 generator에 대한 정보를 이미 학습했기 때문에 compression 과정에서도 그대로 $$D'$$를 사용합니다. <br>
    실제로 random initialize된 $$D$$를 사용했을 때는 훈련이 unstable하고, image의 품질 저하가 일어났습니다. 따라서 pre-trained $$D'$$에 fine-tuning 합니다. <br>
    Objective function은 일반적인 CGAN과 동일하게 사용합니다. <br>

    $$
    \begin{align}
    \mathcal{L}_{CGAN} = \mathbb{E}_{x,y}[\log D'(x,y)] + \mathbb{E}_{x}[\log(1 - D'(x,G'(x)))] \\
    \end{align}
    $$
    
    Model compression에는 output layer의 logit의 분포를 맞추는 knowledge distillation이 대체적으로 많이 쓰입니다. 하지만 CGAN에서는 output이 확률의 분포라기보다 deterministic한 image이기 때문에 distillation하기가 쉽지 않습니다. <br>
    특히 paired dataset으로 훈련된 경우 GT와 generated image의 차이가 많이 없기 때문에 더 잘 동작하지 않습니다. 따라서 teacher $$G$$의 intermediate layer에 대해서 matching을 진행합니다. <br>
    Objective function은 <br>

    $$
    \begin{align}
    \mathcal{L}_{distill} = \sum^{T}_{t=1} \parallel G_{t}(x)-f_{t}(G'_{t}(x))\parallel_{2} \\
    \end{align}
    $$
    
    T는 layer의 개수를 의미하고, $$f_{t}$$는 teacher model에서 student model로 channel의 개수를 mapping하는 1x1 convolution layer입니다. $$\mathcal{L}_{distill}$$는 $$G_{t}$$와 $$f_{t}$$를 optimize합니다. <br><br>
    
    전체 objective function은 <br>

    $$
    \begin{align}
    \mathcal{L} = \mathcal{L}_{CGAN} + \lambda_{recon}\mathcal{L}_{recon} + \lambda_{distill}\mathcal{L}_{distill} \\
    \end{align}
    $$

- Efficient Generator Design Space <br>
    Knoewledge에서 architecture의 선택은 중요합니다. GAN에서 단순히 channel을 줄이는 것은 성능이 현저하게 저하되고 compact한 student model을 생성하지 못합니다. 따라서 CGAN에 대한 더 compact한 architecture를 구축하기 위해 NAS를 사용합니다. <br>

    - Convolution decomposition and layer sensitivity <br>
        Generator는 classification과 segmentation model에서 가져온 vanilla CNN인 경우가 많습니다. Depthwise separable convolution은 performance-computation trade-off에서 효율적이고 generator에서도 마찬가지입니다. Decomposition을 모든 layer에 적용하면 성능상 degradation이 일어나기 때문에 모두 적용하지는 않습니다. Resblock의 경우 model에서 가장 많은 computation cost를 차지하고 있지만 decomposition의 영향을 받지 않고, upsampling layer의 경우 영향을 많이 받기 때문에 resblock에 대해서만 decomposition을 진행합니다. <br>

    - Automated channel reduction with NAS <br>
        기존의 사람이 설계한 균일한 channel을 가진 generator는 optimal하지 않습니다. 불필요한 channel을 없애기 위해 automated channel pruning을 사용하여 channel을 선택합니다. <br>

        각 layer는 MAC와 hardware parallelism의 균형을 위해 8배수로 convolution을 선택할 수 있습니다. 그림에서 $$\{C_{1}, C_{2}, \cdots , C_{k}\}$$가 pruning할 layer의 수이고, $$F_{t}$$가 computation constraint일 때, <br>

        $$
        \begin{align}
        \{C_{1}^{*}, C_{2}^{*}, \cdots , C_{k}^{*}\} = {\argmin}_{C_{1}, \cdots , C_{k}} \mathcal{L}, \quad \text{s.t.} \;\; MAC < F_{t}\\
        \end{align}
        $$

        모든 가능한 channel 조합을 보며 $$\mathcal{L}$$을 optimize하여 가장 optimal한 generator를 고르기 위해 훈련합니다. <br>
        K가 증가하면 가능한 channel configuration은 극단적으로 증가하고, 각 configuration에 대해 hyperparamter 설정에도 많은 시간이 필요로 하는 문제가 있다. <br>

- Decouple Training and Search <br>
    위와 같은 문제를 해결하기 위해 one-shot NAS와 같이 training과 architecture search를 decoupling합니다. <br>
    먼저 onec-for-all network를 학습하고, 각 subnetwork 또한 동일하게 훈련되고 독립적으로 동작합니다. Subnetwork는 once-for-all network와 weight를 공유합니다. <br>

    Teacher model의 channel은 $$\{C_{k}^{0}\}_{k=1}^{K}$$로 가정합니다. <br>
    주어진 $$\{C_{k}\}_{k=1}^{K}, \;\; C_{k} \le C_{k}^{0}$$에서 once-for-all에서 해당 tensor에 대한 weight를 추출합니다. <br>

    Training step에서 subnetwork를 random하게 sampling하고, eq.4로 optimize합니다. 처음 여러 channel이 더 자주 update되기 때문에 전체 weight중에서 중요한 역할을 합니다. <br>

    Onece-for-all network가 훈련된 다음에 validation에 대한 subnetwork의 성능으로 가장 optimal한 subnetwork를 찾습니다. Once-for-all은 weight sharing으로 훈련되기 때문에 추가적은 training 없이 optimal network를 선택할 수 있고, 성능 향상을 위해 fine-tuning도 진행합니다. <br>


#### Experiments
- Models, Datasets, Evaluation Metrics <br>
    - Models <br>
        GAN compression의 generality를 입증하기 위해, unpaired model인 CycleGAN, paired image translation model인 pix2pix, semantic label에 대한 translation model인 GauGAN에 대해 실험을 진행합니다. <br>
        Pix2pix의 경우 UNet을 ResNet으로 교체하여 사용합니다. <br>

    - Dataset <br>


    - Evaluation metrics <br>

- Result <br>
    - Quantitative Result <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_quantitative_result.jpg" width="600"></center>
        <center><em>Fig n.</em></center>
        </p>


    - Performance vs Computation Trade-off <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_trade_off.jpg" width="400"></center>
        <center><em>Fig n.</em></center>
        </p>


    - Qualitative Result <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_qualitative_result.jpg" width="600"></center>
        <center><em>Fig n.</em></center>
        </p>


    - Accelerate Inference on Hardware <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_hardware_inference.jpg" width="350"></center>
        <center><em>Fig n.</em></center>
        </p>

- Ablation study <br>
    - Advantage of unpaired-to-paired transform <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_pseudo_advantage.jpg" width="400"></center>
        <center><em>Fig n.</em></center>
        </p>

    - Effectiveness of convolution decomposition <br>
        <p>
        <center><img src="../images/GAN_compression/Compression_decomposition_performance.jpg" width="400"></center>
        <center><em>Fig n.</em></center>
        </p>

#### Reference 
