function kmp_setup_octave(referenceTimes, referenceMean, referenceCovariance, queryTimes)
% Store one configuration in the persistent Octave worker.
global KMP_REF_TIMES KMP_REF_MEAN KMP_REF_COV KMP_QUERY_TIMES
KMP_REF_TIMES = referenceTimes(:)';
KMP_REF_MEAN = referenceMean;
KMP_REF_COV = referenceCovariance;
KMP_QUERY_TIMES = queryTimes(:)';
end
