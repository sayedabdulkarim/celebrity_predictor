#### FOR getting filenames.pkl ####

# import os
# import pickle
#
# actors = os.listdir('data')
# # print(actors)
#
# fileNames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data', actor)):
#         fileNames.append(os.path.join('data', actor, file))
#
# # print(fileNames)
# # print(len(fileNames), ' length')
#
# # we will dump it to binary obj
# pickle.dump(fileNames, open('filenames.pkl', 'wb'))