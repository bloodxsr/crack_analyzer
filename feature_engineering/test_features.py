from feature_engineering.extractor import extract_features

features = extract_features("E:\\CRACK_AI\\output\\images\\test.png")

for k,v in features.items():
    print(k, ":", v)
