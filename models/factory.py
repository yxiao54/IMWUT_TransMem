# models/factory.py

from models.mymodel import (
    ReSPIRE,
    BaselineNet,
    ConcatNet,
    SelfAttnNet,
    CrossAttnNet,
    MoENet,
    DecisionScaleOnlyNet,
    PhysOnlyNet,
    PersonalizedOnlyNet,
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
    if embedding_mode=='small':
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
        model = ReSPIRE(input_dim, mem_input_dim, hidden, num_class, drop,groups=groups)
        if baseline_ckpt is not None:
            model.load_baseline_from_ckpt(baseline_ckpt)

    elif model_name == "moe":
        model = MoENet(input_dim, mem_input_dim, hidden, num_class, drop)

    elif model_name == "decisionscaleonly":
        model = DecisionScaleOnlyNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    elif model_name == "physonly":
        model = PhysOnlyNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )

    elif model_name == "personalizedonly":
        model = PersonalizedOnlyNet(
            input_dim, mem_input_dim, hidden, num_class, drop, **kwargs
        )
    
    elif model_name in ["irm",'vrex','dro']:
        model = EnvInvariantNet(input_dim, mem_input_dim, hidden, num_class, drop)
   


    else:
        raise ValueError(f"Unknown model_name: {model_name}")



    return model
def get_sklearn_model(name, trial=None):
    name = name.lower()

    if name == "svm":
        # === Paper-aligned Gaussian SVM ===
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",          # ? Gaussian / RBF
            probability=True,
            class_weight="balanced"
        )

    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean"
        )

    else:
        raise ValueError(f"Unknown sklearn model: {name}")
