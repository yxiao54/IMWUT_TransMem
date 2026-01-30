# models/factory.py


from models.newmodel import (ConcatNet, BaselineNet, SelfAttnNet, CrossAttnNet, TransMemNet, MoENet, EnsembleNet)
from models.baseline import (
 
    ProfileBiasNet,
    DecisionScaleOnlyNet,
    PhysGateNet,
    PersonalizedOnlyNet,
    FixedFusionNet,
    EnvInvariantNet,
)


def get_model(
    model_name: str,
    input_dim: int,
    num_class: int = 2,
    baseline_ckpt: str = None,
    embedding_mode: str='old',
    groups: int=16,
    hidden: int=64,
    **kwargs,
):
    """
    Unified model factory.
    """
    model_name = model_name.lower()

    
    drop = kwargs.get("drop", 0.2)
    if embedding_mode=='old':
        mem_input_dim =  3072
    elif embedding_mode=='small':
        mem_input_dim =  3072
    elif embedding_mode=='large':
        mem_input_dim =  3072*2
    else:
         mem_input_dim =  3072


    if model_name == "concate":
        model = ConcatNet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "baseline":
        model = BaselineNet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "selfattn":
        model = SelfAttnNet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "crossattn":
        model = CrossAttnNet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "ours":
        model = TransMemNet(input_dim, mem_input_dim, hidden, num_class, drop,groups=groups)
        if baseline_ckpt is not None:
            model.load_baseline_from_ckpt(baseline_ckpt)

            
    elif model_name == "moe":
        model = MoENet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "ensemble":
        model = EnsembleNet(input_dim, mem_input_dim, hidden, num_class, drop)
    

    elif model_name == "profilebias":
        model = ProfileBiasNet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "decisionscaleonly":
        model = DecisionScaleOnlyNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    elif model_name == "physgate":
        model = PhysGateNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    elif model_name == "personalizedonly":
        model = PersonalizedOnlyNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    elif model_name == "fixedfusion":
        model = FixedFusionNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    
    elif model_name in ["irm",'vrex','dro']:
        model = EnvInvariantNet(input_dim, mem_input_dim, hidden, num_class, drop)
   


    else:
        raise ValueError(f"Unknown model_name: {model_name}")



    return model
