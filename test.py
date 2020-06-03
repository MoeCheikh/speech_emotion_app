import os
from utils import convert, extract_feature


list_of_files = []
for file in os.listdir("./test"):
    list_of_files.append(f"{os.getcwd()}/test/{file}")

def test_conversion():
    print("Testing Wav Conversion...")
    result = convert(list_of_files[1])
    if isinstance(result,str):
        print("WAV CONVERSION PASSED")
        os.remove(result)
    else:
        print("WAV CONVERSION FAILED")


def test_feature_extraction():
    print("Testing Feature Extraction...")
    result = extract_feature(list_of_files[0])
    if len(result) > 1:
        print(result)
        print("FEATURE EXTRACTION PASSED")
    else:
        print("FEATURE EXTRACTION FAILED")

test_feature_extraction()
test_conversion()
