import word2vec as w2v

def validate_model(model_fp):
    print("Start validation")

    model = w2v.load(model_fp)


    return model #TODO remove return value, only used for development
