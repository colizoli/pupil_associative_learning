# pupil_preprocessing

General preprocessing scripts for pupillometry (EyeLink eye-tracker)

For EyeLink, need to have the edf2asc executable, install desktop application. <br>
If not using EyeLink, can start from preprocessing_functions_control_exp.preprocess_pupil(), with self.pupil as time series.

For reading trials:
Trials are marked by a 1 at the index of the onset of each event with respect to the entire pupil time series.

