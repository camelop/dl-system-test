import os
import sys

python_cmd = "python"
testcase_dir = "testcase"
tests = [
    ["adder",       "1_adder.py"],
    ["initializer", "2_init.py"],
    ["assign",      "3_assign.py"],
    ["context",     "4_context.py"],
    ["autodiff",    "5_mnist_grad.py"],
    ["optimizer",   "6_mnist_optimizer.py"],
    ["multilayer perceptron", "7_ml_perceptron.py"],
    ["adam optimizer",        "8_adam.py"],
    ["CNN 1",       "9_cnn_1.py"],
    ["CNN 2",       "10_cnn_2.py"]
]


def main(model_name):
    for i, item in enumerate(tests):
        test_name = item[0]
        file_name = item[1]

        print("test %d %s" % (i, test_name))
        path = "C:\\Users\\lxy98\\Documents\\MyGitProjects\\test_your_own-dlsys\\testcase_my\\"
        ret = os.system("%s %s" % (python_cmd, path + file_name))
        if ret != 0:
            print("Failed")
            exit(0)
        else:
            print("Accept")
        print("")
    print("Pass all!")


if __name__ == "__main__":
    main('tensorwolf')
