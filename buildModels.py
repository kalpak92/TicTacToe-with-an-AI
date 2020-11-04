from regressor_final import execute_regressor
from classifier_final import execute_classifier

if __name__ == '__main__':
    split_size = 0.8
    print("********** Classifiers **********")
    print("--> Final move dataset <--")
    final_dataset_path = "dataset/tictac_final.txt"
    execute_classifier(final_dataset_path, split_size, "Final move dataset - ",
                       k_range=100, fraction_10th=False)
    print("---" * 20)

    print("--> Intermediate board's optimal move dataset <--")
    final_dataset_path = "dataset/tictac_single.txt"
    execute_classifier(final_dataset_path, split_size, "Single Classifier dataset - ",
                       k_range=100, fraction_10th=False)
    print("---" * 20)

    print("---> Running the Models on 10% of the dataset <---")
    print("--> Final move dataset <--")
    final_dataset_path = "dataset/tictac_final.txt"
    execute_classifier(final_dataset_path, split_size, "Final move dataset(10_th_Fraction) - ",
                       k_range=10, fraction_10th=True)
    print("---" * 20)

    print("--> Intermediate board's optimal move dataset <--")
    final_dataset_path = "dataset/tictac_single.txt"
    execute_classifier(final_dataset_path, split_size, "Single Classifier dataset(10_th_Fraction) - ",
                       k_range=10, fraction_10th=True)
    print("---" * 20)

    print("********** Regressors **********")
    final_multi_dataset_path = "dataset/tictac_multi.txt"
    execute_regressor(final_multi_dataset_path, split_size)