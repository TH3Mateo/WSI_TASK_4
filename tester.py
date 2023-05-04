# unacc = [1 if x == 1 else 0 for x in effect]
# acc = [1 if x == 2 else 0 for x in effect]
# good = [1 if x == 3 else 0 for x in effect]
# vgood = [1 if x == 4 else 0 for x in effect]
#
# unacc_exp = [1 if x == 1 else 0 for x in expected]
# acc_exp = [1 if x == 2 else 0 for x in expected]
# good_exp = [1 if x == 3 else 0 for x in expected]
# vgood_exp = [1 if x == 4 else 0 for x in expected]
#
#
# # print(sum(diff))
# # print(b_diff)
# # print("ACCURACY: ",np.sum(b_diff)/len(b_diff))
# # print()
#
# TP = sum([1 if effect[i] == 2 and expected[i] == 2 else 0 for i in range(len(effect))])
#
# # Calculate false positives (FP)
# FP = sum([1 if effect[i] != 2 and expected[i] == 2 else 0 for i in range(len(effect))])
#
# # Calculate false negatives (FN)
# FN = sum([1 if effect[i] == 2 and expected[i] != 2 else 0 for i in range(len(effect))])
#
# # Calculate true negatives (TN)
# TN = sum([1 if effect[i] != 2 and expected[i] != 2 else 0 for i in range(len(effect))])
#
# # Calculate accuracy, precision, recall and F1 score
#
# print("################################")
# print("DEPTH: ",depth)
# print("TREE COUNT: ",tree_count)
# print("################################")
# accuracy = (TP + TN) / len(effect)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# f1_score = 2 * precision * recall / (precision + recall)
#
# # Print confusion matrix and error table
# print('Confusion matrix:')
# print(f'TP: {TP}, FP: {FP}')
# print(f'FN: {FN}, TN: {TN}')
# print('\nError table:')
# print(f'Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1 score: {f1_score:.2f}')