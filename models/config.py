# Model name -> category directory: ML (machine learning), DL (deep learning), FM (foundation model)
MODEL_CATEGORY = {
    'CSP_LDA': 'ML',
    'Xdawn_LDA': 'ML',
    'PSD_Ridge': 'ML',
    'DE_LDA': 'ML',
    'TRCA': 'ML',
    'PSD_LDA': 'ML',
    'PSD_SVM': 'ML',
    'EEGNet': 'DL',
    'ShallowConv': 'DL',
    'CNNTransformer': 'DL',
    'Deformer': 'DL',
    'Conformer': 'DL',
    'LMDA': 'DL',
    'FBCNet': 'DL',
    'MSCFormer': 'DL',
    'TSception': 'DL',
    'LaBraM': 'FM',
    'BENDR': 'FM',
    'NeuroGPT': 'FM',
    'BrainOmni': 'FM',
    'BrainOmni_B': 'FM',
    'BrainOmni_Tiny': 'FM',
    'BrainOmni_Base': 'FM',
    'LUNA': 'FM',
    'LUNA_Base': 'FM',
    'LUNA_H': 'FM',
    'LUNA_L': 'FM',
    'EEGMamba': 'FM',
    'SingLEM': 'FM',
    'TFMTokenizer': 'FM',
    'CBraMod': 'FM',
    'EEGPT': 'FM',
    'BIOT': 'FM',
    'BIOT_1': 'FM',
    'BIOT_2': 'FM',
    'BIOT_6D': 'FM',
    'BIOT_1D': 'FM',
    'BIOT_2D': 'FM',
}

CATEGORY_NAMES = {
    'ML': 'Machine Learning',
    'DL': 'Deep Learning',
    'FM': 'Foundation Model',
}


def get_model_category(model_name: str) -> str:
    """Return category directory name (ML / DL / FM) for the given model."""
    if model_name not in MODEL_CATEGORY:
        raise ValueError(f"Unknown model name: {model_name}. Known: {list(MODEL_CATEGORY.keys())}")
    return MODEL_CATEGORY[model_name]
