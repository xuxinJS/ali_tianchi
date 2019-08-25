import os
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, StrMethodFormatter, FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from NoduleFinding import NoduleFinding
from tools import csvTools

# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
# TODO
# diameter_mm_label = 'diameter_mm'
diameter_x_label = 'diameterX'
diameter_y_label = 'diameterY'
diameter_z_label = 'diameterZ'
nodule_type = 'label'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.25  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve
bLogPlot = True


def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]

    # get a random list of images using sampling with replacement
    rand_index_im = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]

    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue

        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scanToCandidatesDict[im]), axis=1)

    return candidates


def compute_mean_ci(interp_sens, confidence=0.95):
    sens_mean = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_lb = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_up = np.zeros((interp_sens.shape[1]), dtype='float32')

    Pz = (1.0 - confidence) / 2.0

    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:, i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz * len(vec))]
        sens_up[i] = vec[math.floor((1.0 - Pz) * len(vec))]

    return sens_mean, sens_lb, sens_up


def computeFROC_bootstrap(FROCGTList, FROCProbList, FPDivisorList, FROCImList, excludeList,
                          numberOfBootstrapSamples=1000, confidence=0.95):
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)

    fps_lists = []
    sens_lists = []
    thresholds_lists = []

    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)

    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:, i:i + 1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid], candidate), axis=1)

    for i in range(numberOfBootstrapSamples):
        print
        'computing FROC: bootstrap %d/%d' % (i, numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict, FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0, :], btpsamp[1, :], len(FROCImList_np), btpsamp[2, :])

        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)

    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fps)), dtype='float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i, :] = np.interp(all_fps, fps_lists[i], sens_lists[i])

    # compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence=confidence)

    return all_fps, sens_mean, sens_lb, sens_up


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(
            FROCGTList):  # Handle border case when there are no false positives and ROC analysis give nan values.
        print
        "WARNING, this system has no false positives.."
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds


def evaluateCAD(seriesUIDs, results, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False, numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results: infer results with special type
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    nodOutputfile = open(os.path.join(outputDir, 'CADAnalysis.txt'), 'w')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    # results = csvTools.readCSV(results_filename)

    allCandsCAD = {}

    for seriesuid in seriesUIDs:

        # collect candidates from result file
        nodules = {}
        header = results[0]

        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.items():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.items():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2

        # print
        # 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules  # 遍历推理的结果文件，将某个seriesuid的所有结果合在一起，和seriesuid做成键值对

    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1
            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            # TODO
            # diameter = float(noduleAnnot.diameter_mm)
            diameter_X = float(noduleAnnot.diameter_x)
            diameter_Y = float(noduleAnnot.diameter_y)
            diameter_Z = float(noduleAnnot.diameter_z)
            if diameter_X <= 2.0:
                diameter_X = 2.0
            if diameter_Y <= 2.0:
                diameter_Y = 2.0
            if diameter_Z <= 2.0:
                diameter_Z = 2.0
            # radiusSquared = pow((diameter / 2.0), 2.0)
            diameter_X_cal = diameter_X / 2
            diameter_Y_cal = diameter_Y / 2
            diameter_Z_cal = diameter_Z / 2

            found = False
            noduleMatches = []
            # print(candidates)
            for key, candidate in candidates.items():
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                # TODO
                # dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
                x_dist = abs(x - x2)
                y_dist = abs(y - y2)
                z_dist = abs(z - z2)
                # if dist < radiusSquared:
                if x_dist < diameter_X_cal and y_dist < diameter_Y_cal and z_dist < diameter_Z_cal:
                    if (noduleAnnot.state == "Included"):
                        found = True
                        noduleMatches.append(candidate)
                        if key not in candidates2.keys():
                            print
                            "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (
                                str(candidate.id), seriesuid, str(noduleAnnot.id))
                        else:
                            del candidates2[key]
                    elif (noduleAnnot.state == "Excluded"):  # an excluded nodule
                        if bOtherNodulesAsIrrelevant:  # delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                                    seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ,
                                    str(candidate.id),
                                    float(candidate.CADprobability)))
                                del candidates2[key]
            if len(noduleMatches) > 1:  # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            if noduleAnnot.state == "Included":
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0)
                    FROCProbList.append(float(maxProb))
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    # TODO
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%.9f" % (
                        seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                        float(noduleAnnot.diameter_x), float(noduleAnnot.diameter_y), float(noduleAnnot.diameter_z),
                        str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1
                else:
                    candFNs += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
                    # TODO
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%s" % (
                        seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                        float(noduleAnnot.diameter_x), float(noduleAnnot.diameter_y), float(noduleAnnot.diameter_z),
                        int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s\n" % (
                        seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                        float(noduleAnnot.diameter_x), float(noduleAnnot.diameter_y), float(noduleAnnot.diameter_z),
                        str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.items():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id),
                float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(
            FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write(
        "    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write(
        "    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)

    if performBootstrapping:
        fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(FROCGTList, FROCProbList,
                                                                                 FPDivisorList, seriesUIDs, excludeList,
                                                                                 numberOfBootstrapSamples=numberOfBootstrapSamples,
                                                                                 confidence=confidence)

    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))

    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

    sens_itp = np.interp(fps_itp, fps, sens)

    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':')  # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':')  # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))

        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.25, 0.5, 0.75, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks([0.25, 0.5, 0.75, 1, 2, 4, 8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

        # Calculate  Sensitivity在1/8, 1/4, 1/2, 1, 2, 4和8一共7个不同的误报情况下的平均值作为其中一种病灶的得分
        cal_score_list = [0.25, 0.5, 0.75, 1, 2, 4, 8]
        scale = (FROC_maxX - FROC_minX) / 10000
        all_score = 0
        for i in cal_score_list:
            index = int((i - FROC_minX) / scale)
            if index < 0:
                index = 0
            elif index > 9999:
                index = 9999
            print("%.2f %f" % (i, sens_bs_mean[index]))
            all_score += sens_bs_mean[index]
        score = all_score / (len(cal_score_list))
        return score
    else:
        return 0

    # return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)


def getNodule(annotation, header, state=""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    # TODO
    if diameter_x_label in header:
        nodule.diameter_x = annotation[header.index(diameter_x_label)]
    if diameter_y_label in header:
        nodule.diameter_y = annotation[header.index(diameter_y_label)]
    if diameter_z_label in header:
        nodule.diameter_z = annotation[header.index(diameter_z_label)]

    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]

    if not state == "":
        nodule.state = state
    if nodule_type in header:
        nodule.noduleType = annotation[header.index(nodule_type)]

    return nodule


def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    # allNodules：将annotations和annotations_excluded里面的同一seriesuid号的数据合在一起
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0

    for seriesuid in seriesUIDs:
        # print
        # 'adding nodule annotations: ' + seriesuid

        nodules = []
        numberOfIncludedNodules = 0

        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1

        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Excluded")
                nodules.append(nodule)

        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)

    # print
    # 'Total number of included nodule annotations: ' + str(noduleCount)
    # print
    # 'Total number of nodule annotations: ' + str(noduleCountTotal)
    return allNodules


# def collect(annotations_filename, annotations_excluded, seriesUIDs):
#     annotations = csvTools.readCSV(annotations_filename)
#     annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
#     seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)
#
#     seriesUIDs = []
#     for seriesUID in seriesUIDs_csv:
#         seriesUIDs.append(seriesUID[0])
#
#     allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)
#
#     return (allNodules, seriesUIDs)


def noduleCADEvaluation(annotations_filename, results_filename, save_name, outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''

    # print
    # annotations_filename
    annotations = csvTools.readCSV(annotations_filename)
    header = annotations[0]
    annotations_excluded = []
    annotations_excluded.append(header)
    seriesUIDs = []
    for annotation in annotations[1:]:
        _seriesUID = annotation[header.index(seriesuid_label)]
        if _seriesUID not in seriesUIDs:
            seriesUIDs.append(_seriesUID)

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)

    # (allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)

    # 分类别进行计算FROC值
    nodule_type_list = ['1', '5', '31', '32']
    nodules_list = []
    seriesUIDs_list = []
    for _nodule_type in nodule_type_list:
        name_nodules = 'nodules_' + _nodule_type
        name_seriesUIDs = 'seriesUIDs_' + _nodule_type
        locals()[name_nodules] = dict()
        locals()[name_seriesUIDs] = list()

        for seriesUID, nodule_obj_list in allNodules.items():
            _nodule_list = []
            for nodule_obj in nodule_obj_list:
                if nodule_obj.noduleType == _nodule_type:
                    _nodule_list.append(nodule_obj)
                    if seriesUID not in locals()[name_seriesUIDs]:
                        (locals()[name_seriesUIDs]).append(seriesUID)
            if len(_nodule_list):
                (locals()[name_nodules])[seriesUID] = _nodule_list
        nodules_list.append(locals()[name_nodules])
        seriesUIDs_list.append(locals()[name_seriesUIDs])

    all_infer_results = csvTools.readCSV(results_filename)
    header = all_infer_results[0]
    results_list = []
    for _nodule_type in nodule_type_list:
        results_nodules = 'results' + _nodule_type
        locals()[results_nodules] = list()
        (locals()[results_nodules]).append(header)
        for result in all_infer_results[1:]:
            if result[header.index(nodule_type)] == _nodule_type:
                (locals()[results_nodules]).append(result)
        results_list.append(locals()[results_nodules])

    all_score = 0.0
    mean_score = 0.0
    log = open(save_name, 'w')
    for index, _nodule_type in enumerate(nodule_type_list):
        score_nodule = 'score_' + _nodule_type
        locals()[score_nodule] = evaluateCAD(seriesUIDs_list[index], results_list[index], outputDir,
                                             nodules_list[index],
                                             os.path.splitext(os.path.basename(results_filename))[0],
                                             maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                                             numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)
        log.write("%s:%f\n" % (score_nodule, locals()[score_nodule]))
        print("%s:%f" % (score_nodule, locals()[score_nodule]))
        all_score += locals()[score_nodule]
    mean_score = all_score / len(nodule_type_list)
    log.write("mean score:%f\n" % mean_score)
    print("mean score:%f" % mean_score)
    log.close()


if __name__ == '__main__':
    annotations_filename = sys.argv[1]
    results_filename = sys.argv[2]
    results_save_filename = sys.argv[3]
    outputDir = sys.argv[4]
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    # execute only if run as a script
    noduleCADEvaluation(annotations_filename, results_filename, results_save_filename, outputDir)
    print("Finished!")
