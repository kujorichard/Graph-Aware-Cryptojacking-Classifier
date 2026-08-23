from .dataset_builder import DatasetBuilder
from .vocabulary import Vocabulary
from .loader import GraphLoader
from .extractors import (extract_binary_cfg_ngrams, extract_binary_dfg_ngrams,)

def main():

    binary = GraphLoader.load("D:\\BSCS\\Cryptojacking Detection Thesis\\Data Collection V2\\graph_ngrams\\dataset\\benign\\example2.exe.json")
    binary1 = GraphLoader.load("D:\\BSCS\\Cryptojacking Detection Thesis\\Data Collection V2\\graph_ngrams\\dataset\\malware\\example1.exe.json")

    print("=" * 60)
    print(f"Binary : {binary.binary_name}")
    print(f"Label  : {binary.label_name}")
    print(f"Functions : {len(binary.functions)}")
    print("=" * 60)

    for func in binary.functions[:5]:

        print(f"\nFunction : {func.name}")
        print(f"Entry    : {func.entry}")

        print(f"Instructions : {len(func.instructions)}")
        print(f"CFG Blocks   : {len(func.cfg_nodes)}")
        print(f"CFG Edges    : {len(func.cfg_edges)}")
        print(f"DFG Edges    : {len(func.dfg_edges)}")

        if func.instructions:
            first = func.instructions[0]

            print("\nFirst instruction")
            print(
                f"  [{first.idx}] "
                f"{first.address} "
                f"{first.mnemonic} "
                f"{first.operands}"
            )


    # --------------------- DatasetBuilder Temporary testing --------------------------#
    # binaries = [binary, binary1]

    # vocabulary = Vocabulary()

    # for binary in binaries:

    #     dfg = extract_binary_dfg_ngrams(binary, 3)
    #     cfg = extract_binary_cfg_ngrams(binary, 3)

    #     shingles = dfg + cfg

    #     vocabulary.add_shingles(
    #         binary.binary_name,
    #         shingles,
    #     )

    # print("Vocabulary size:", len(vocabulary.entries))


    # builder = DatasetBuilder(vocabulary)

    # X, y = builder.build_feature_matrix(binaries)

    # print(f"Number of samples: {len(X)}")
    # print(f"Vocabulary size: {len(X[0])}")
    # print(f"Labels: {y}")

    # print(f"Length of first feature vector: {len(X[0])}")

    # non_zero = sum(
    #     1
    #     for value in X[0]
    #     if value > 0
    # )

    # print(f"Non-zero features: {non_zero}")

    # print("First 20 features:")
    # print(X[0][:20])

    # print(sum(X[0]))
    # print(sum(X[1]))
    #--------------------------------------------#


    # --------------------- Displaying CFG/DFG n-grams Temporary testing --------------------------#
    cfg = extract_binary_cfg_ngrams(binary, 3)
    dfg = extract_binary_dfg_ngrams(binary, 3)

    print(dfg)



    #--------------------------------------------#

if __name__ == "__main__":
    main()
