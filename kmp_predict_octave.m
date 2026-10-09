function prediction = kmp_predict_octave(kh, lambda)
% KMP posterior mean for state [x,y,vx,vy] using a persistent problem cache.
global KMP_REF_TIMES KMP_REF_MEAN KMP_REF_COV KMP_QUERY_TIMES
if isempty(KMP_REF_TIMES)
    error('kmp_predict_octave:NoProblem', 'Call kmp_setup_octave first.');
end

kernel = extendedKernel(KMP_QUERY_TIMES, KMP_REF_TIMES, kh);
systemMatrix = extendedKernel(KMP_REF_TIMES, KMP_REF_TIMES, kh);
for i = 1:numel(KMP_REF_TIMES)
    rows = (i - 1) * 4 + (1:4);
    systemMatrix(rows, rows) = systemMatrix(rows, rows) + lambda * KMP_REF_COV(:, :, i);
end
alpha = systemMatrix \ KMP_REF_MEAN(:);
prediction = reshape(kernel * alpha, 4, []);
end

function matrix = extendedKernel(queryTimes, referenceTimes, kh)
fdStep = 0.001;
delta = queryTimes(:) - referenceTimes(:)';
k00 = exp(-kh * delta.^2);
k0dTemp = exp(-kh * (delta - fdStep).^2);
kd0Temp = exp(-kh * (delta + fdStep).^2);
k0d = (k0dTemp - k00) / fdStep;
kd0 = (kd0Temp - k00) / fdStep;
kdd = (2 * k00 - k0dTemp - kd0Temp) / fdStep^2;

nq = numel(queryTimes);
nr = numel(referenceTimes);
matrix = zeros(4 * nq, 4 * nr);
for axis = 1:2
    qp = (0:nq-1) * 4 + axis;
    qv = qp + 2;
    rp = (0:nr-1) * 4 + axis;
    rv = rp + 2;
    matrix(qp, rp) = k00;
    matrix(qp, rv) = k0d;
    matrix(qv, rp) = kd0;
    matrix(qv, rv) = kdd;
end
end
