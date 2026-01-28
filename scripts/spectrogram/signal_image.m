% Read the spectrogram image
spectrogramImage = imread('gen_images2.png');

% Convert the image to grayscale if it is not already
grayImage = rgb2gray(spectrogramImage);

% Perform a 2D FFT on the grayscale image
fftResult = fft2(double(grayImage));

% Shift zero-frequency component to the center of the spectrum
fftShifted = fftshift(fftResult);

% Calculate the magnitude
magnitude = abs(fftShifted);

% Find the frequencies corresponding to each FFT bin
samplingFrequency = 1; % Assuming a sampling frequency of 1 Hz for simplicity
frequencyResolution = samplingFrequency / size(grayImage, 2); % Frequency resolution per bin
frequencies = -samplingFrequency/2 : frequencyResolution : samplingFrequency/2 - frequencyResolution;

% Find peaks in the magnitude spectrum
[peaks, locs] = findpeaks(magnitude(:), 'MinPeakHeight', mean(magnitude(:)), 'MinPeakDistance', 10);

% Check if locs is empty
if isempty(locs)
    disp('No peaks found in the magnitude spectrum.');
    return;
end

% Ensure locs is within valid range
valid_locs = locs(locs >= 1 & locs <= length(frequencies));

% Calculate bandwidth in MHz
bandwidth_MHz = max(frequencies(valid_locs)) - min(frequencies(valid_locs));

% Calculate sub-carrier spacing in kHz
subcarrier_spacing_kHz = mean(diff(frequencies(valid_locs))) * 1e-3;

% Calculate center frequency
center_frequency = frequencies(ceil(length(frequencies)/2));

% Analyze modulation (additional steps required)
% You'll need to implement modulation analysis based on your signal characteristics.
% This may involve techniques such as constellation diagram analysis or autocorrelation.

% Display results
disp(['Center Frequency: ', num2str(center_frequency), ' Hz']);
disp(['Bandwidth: ', num2str(bandwidth_MHz), ' MHz']);
disp(['Sub-Carrier Spacing: ', num2str(subcarrier_spacing_kHz), ' kHz']);