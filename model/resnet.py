import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self,
                 model: str = '34',
                 num_classes: int = 1000,
                 use_residual: bool = True,
                 use_transformation_in_shortcut: str = 'none',
                 config: dict = None
                ):
        """
        num_layers(int): 사용할 layer의 개수. 현재는 논문에 따라 18, 34, 50, 101, 152만 구현. 다른 형식을 위해서는 self.config에 추가 필요.
        num_classes(int): 분류할 라벨의 개수
        use_residual(int): shortcut connection을 사용하여 residual mapping할 지의 여부. False로 설정할 경우 논문의 Plain Network로 기능
        use_transformation_in_shorcut(str): shortcut connection을 할 때, input에 1x1 conv를 사용하여 선형변환할지의 여부 (아래의 세 가지 param)
            'none': 논문의 A방식 --> identity mapping - 1x1 conv를 사용하지 않고 같은 값을 더하며, 차원이 달라질 경우 추가된 차원에 zero padding
            'part': 논문의 B방식 --> input과 output의 차원이 같지 않을 경우에만 1x1 conv를 사용하여 input의 차원을 변환
            'all': 논문의 C방식 --> input과 output의 차원의 관계없이, 모든 경우에 1x1 conv를 사용하여 input을 변환
        """
        super(ResNet, self).__init__()  # nn.Module의 생성자 호출

        self.model = str(model)
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.use_transformation_in_shortcut = use_transformation_in_shortcut

        ### 모델 아키텍쳐를 config에 선언 ###
        # config 입력이 없을 경우 default config 적용
        if not config:
            self.config = {
                'channel_size': {
                                    'conv1': 3,
                                    'conv2': 64, 
                                    'conv3': 128, 
                                    'conv4': 256, 
                                    'conv5': 512,
                                },
                'architecture':{
                    '18': {
                        'use_bottleneck': False,
                        'repeat_by_block': [2, 2, 2, 2],
                    },
                    '34': {
                        'use_bottleneck': False,
                        'repeat_by_block': [3, 4, 6, 3],
                    },
                    '50': {
                        'use_bottleneck': True,
                        'repeat_by_block': [3, 4, 6, 3],
                    },
                    '101': {
                        'use_bottleneck': True,
                        'repeat_by_block': [3, 4, 23, 3],
                    },
                    '152': {
                        'use_bottleneck': True,
                        'repeat_by_block': [3, 8, 36, 3],
                    },
                },
            }
        else:
            self.config = config

        ##### 입력값 validation #####
        self.available_model_list = self.config['architecture'].keys()
        if self.model not in self.available_model_list:
            raise Exception(f"지정되지 않은 모델 {self.model}를 입력했습니다.가능한 모델은 {list(self.available_model_list)}입니다.")
            
        use_transformation_in_shortcut_param_list = ['none', 'part', 'all']
        if self.use_transformation_in_shortcut not in use_transformation_in_shortcut_param_list:
            raise Exception(f"use_transformation_in_short 파라미터에 대해 {self.use_transformation_in_shortcut}은 잘못된 값입니다. 가능한 값은 {use_transformation_in_shortcut_param_list} 중 하나 입니다.")
            

        self.arch_config = self.config['architecture'][self.model]
        self.channel_config = self.config["channel_size"]

        ##### ReLU 선언 #####
        self.relu = nn.ReLU()
        
        ##### conv1 layer 선언 (모든 layer버전에 공통적)#####
        self.conv1 = nn.Conv2d(in_channels=self.channel_config['conv1'], out_channels=self.channel_config['conv2'], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        ##### conv2_x block부터 conv5_x block 선언 #####
        # conv block은 순서대로 self.block_list 변수에 저장
        self.block_list = nn.ModuleList()
        # shortcut connection에 사용되는 1x1 conv layer는 self.transformation_layer_list에 저장
        self.transformation_layer_list = nn.ModuleList()
        # self.arch_config에서 각 block 단계별 횟수를 가져와서 횟수만큼 block 생성
        for i, repeat_num in enumerate(self.arch_config["repeat_by_block"]):
            conv_target = f"conv{i+2}"
            channels = self.channel_config[conv_target]
            for seq in range(1, repeat_num+1):
                if seq == 1: # 각 단계의 block list의 첫번째 block은 stride를 2로 지정하여 feature map을 축소하고, channel을 증가시키기 때문에 is_first 파라미터로 표시
                    is_first=True
                else:
                    is_first=False

                # _create_block 메소드를 사용하여 block 생성 후, self.block_list 변수에 저장
                # block이 단계의 첫번째인지 여부와 bottleneck을 사용하는지 여부에 따라 형태가 달라지기 때문에 _create_block 메소드에 argument로 전달
                block = self._create_block(channels=channels, use_bottleneck=self.arch_config["use_bottleneck"], is_first=is_first)
                self.block_list.append(block)

                # _create_transformation_layer 메소드를 사용하여 shortcut connection에 대한 projection layer 생성 후, self.transformation_list 변수에 저장
                # use_transformation_in_shortcut이 'all'을 경우 모든 block의 short connection에 대한 projection layer가 필요함
                # use_transformation_in_shortcut이 'part'일 경우 input과 output의 차원이 바뀌었을 때만 projection layer을 사용함
                # 특이사항은 conv2_x의 첫번째 block의 경우 bottleneck을 사용하지 않으면 입력과 출력 모두 64차원으로 같은데 비해서,
                # bottleneck을 사용할 경우 입력은 64차원이고 출력은 256차원이 되어서 projection layer가 필요함
                # 결론적으로 'part' 경우 bottleneck을 사용하지 않는 18, 34는 3개의 projection layer를, bottltnectk을 사용하는 50, 101, 152는 4개의 projection layer를 가짐
                if (self.use_transformation_in_shortcut == "part" and is_first == True) or (self.use_transformation_in_shortcut == "all"):
                    if (self.use_transformation_in_shortcut == "part" and channels == 64 and not self.arch_config["use_bottleneck"]):
                        continue
                    transformation_layer = self._create_transformation_layer(channels=channels, use_bottleneck=self.arch_config["use_bottleneck"], is_first=is_first)
                    self.transformation_layer_list.append(transformation_layer)                
        
        ##### classification layer #####
        # avgpool은 모든 layer버전에 공통적임
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # linear layer 선언 - bottleneck을 이용할 경우 마지막 channel은 일반 block 대비 4배 증가함
        last_channels = self.channel_config["conv5"]
        if self.arch_config["use_bottleneck"]:
            last_channels *= 4
        self.linear = nn.Linear(last_channels, self.num_classes)            

    def _create_block(self, channels, use_bottleneck, is_first):
        if use_bottleneck:
            # bottleneck구조를 사용할 경우 input과 output의 채널은 일반 block의 4배임
            augmented_channels = int(channels * 4)
            # 각 단계의 첫번째 block들은 이전 block 대비 차원을 2배로 증가시키고, featuremap size를 반으로 줄임. 따라서 stride를 2로 설정해야함
            if is_first:
                first_stride = 2
                # conv2_1를 제외한 각 conv 단계의 첫 layer들은 전 단계의 conv block으로부터 절반의 차원을 지닌 feature를 입력받음
                if channels != 64:
                    first_inchannels = int(augmented_channels / 2)
                else:
                    # 단 conv2_1인 경우 conv1으로부터 같은 차원의 입력을 받음
                    first_inchannels = 64
                    # conv2_1의 경우 앞의 maxpooling으로부터 feature의 size가 축소되었기 때문에 stride를 1로 지정
                    first_stride = 1
            # 각 단계의 첫번째 block을 제외하면 이전 block과 같은 차원, 같은 사이즈의 feature을 입력 받음
            else:
                first_inchannels = augmented_channels
                first_stride = 1
    
            block = nn.Sequential(
                nn.Conv2d(in_channels=first_inchannels, out_channels=channels, kernel_size=1, stride=first_stride),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=channels, out_channels=augmented_channels, kernel_size=1),
                nn.BatchNorm2d(augmented_channels)
            )

        # bottleneck구조를 사용하지 않는 경우에도 channel의 차원이 낮다는 점을 제외하면 bottleneck구조와 같은 논리에 따름
        elif not use_bottleneck:
            if is_first:
                first_inchannels = int(channels/2)
                first_stride = 2
                if channels == 64:
                    first_inchannels = 64
                    first_stride = 1
            else:
                first_inchannels = channels
                first_stride = 1
                
            block = nn.Sequential(
                nn.Conv2d(in_channels=first_inchannels, out_channels=channels, kernel_size=3, padding=1, stride=first_stride),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
            )
            
        return block

    def _create_transformation_layer(self, channels, use_bottleneck, is_first):
        if use_bottleneck:
            # bottleneck 구조를 사용할 경우 일반 block에 대비하여 4배 큰 차원의 입력과 출력을 가짐
            augmented_channels = channels * 4
            if is_first:
                # conv2_1을 제외한 첫번째 bottleneck block의 경우 전 단계의 block으로부터 출력의 1/2 차원의 입력을 받음
                in_channels = int(augmented_channels / 2)
                out_channels = augmented_channels
                stride = 2
                # conv2_1의 경우 입력은 64차원이고 출력은 256차원으로 다른 convX_1 block과 다른 특성을 지님
                if channels == 64:                    
                    in_channels = channels
                    out_channels = augmented_channels
                    stride = 1
            else:
                in_channels = augmented_channels
                out_channels = augmented_channels
                stride = 1            
        else:
            # bottltneck을 사용하지 않을 경우 bottleneck을 사용할 때와 다르게 conv2_1의 입력과 출력이 64차원으로 동일함 
            # 다른 convX_1 block의 경우 bottleneck 사용여부와 무관하게 이전 대비 출력은 두 배고 feature map 크기는 반이라는 동일한 특성을 지님
            if is_first and channels != 64: 
                in_channels = int(channels / 2)
                out_channels = channels
                stride = 2
            else:
                in_channels = channels
                out_channels = channels
                stride = 1            
        transformation_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        return transformation_layer        
        
    def forward(self, x):
        ### conv1 ###
        # conv1은 layer 개수에 상관없이 모든 ResNet 버전에 공통적임
        out = self.conv1(x)
        out = self.maxpool(out)

        ### conv2_x ~ conv5_x ###
        transformation_layer_index = 0
        # conv2_x부터 conv5_x까지 모든 block은 self.block_List에 저장되어 있기 때문에 이를 순회하면서 forward함
        for layer in self.block_list:
            # residual mapping을 위해 입력값을 저장함
            residual_value = out.clone()
            out = layer(out)
            # residual function 적용
            # 1x1 conv projection을 활용한 residual mapping 방식을 적용할 경우 self.transformation_layer_list에 
            # layer들을 순서대로 저장해 놓고 불러와서 사용
            if self.use_residual:
                _, residual_dimension, residual_height , residual_width = residual_value.shape
                _, out_dimension, out_height, out_width = out.shape
                # 논문의 C('all') 방식의 경우 input과 output의 차원에 관계없이 1x1 conv를 사용하여 차원 변환
                if self.use_transformation_in_shortcut == 'all':
                    residual_value = self.transformation_layer_list[transformation_layer_index](residual_value)
                    transformation_layer_index += 1
                    if residual_value.shape != out.shape:
                        raise Exception(f"""
                        use_transformation_in_shrot이 'all'일 때 프로세스가 잘못되었습니다.
                        프로세스 이후 input과 output의 차원이 다릅니다.
                        input차원: {residual_value.shape}, output차원: {out.shape}
                        """)
                    
                # 논문에서 input과 output의 차원이 다를 경우 두 가지 방식이 존재함
                # A('none') --> 증가한 차원에 zero padding
                # B('part') --> 증가한 차원에 맞추어 1x1 conv를 이용하여 차원 변환                
                if residual_dimension != out_dimension:
                    if self.use_transformation_in_shortcut == 'none':
                        residual_value = F.pad(residual_value, pad=(0, 0, 0, 0, 0, abs(out_dimension - residual_dimension)))
                        # 크기가 바뀌었을 경우에는 F.interpolate 메소드를 사용하여, feature map에서 중간값 추출
                        if residual_height != out_height:
                            residual_value = F.interpolate(residual_value, out_height)
                    elif self.use_transformation_in_shortcut == 'part':
                        residual_value = self.transformation_layer_list[transformation_layer_index](residual_value)
                        transformation_layer_index += 1
                        if residual_value.shape != out.shape:
                            raise Exception(f"""
                            use_transformation_in_shrot이 'part'일 때 프로세스가 잘못되었습니다.
                            프로세스 이후 input과 output의 차원이 다릅니다.
                            input차원: {residual_value.shape}, output차원: {out.shape}
                            """)
                out += residual_value
            # residual function 적용 이후 ReLU actionvation 적용
            out = self.relu(out)
        
        # feature 추출 이후
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        out = F.softmax(out, dim=-1)
        return out