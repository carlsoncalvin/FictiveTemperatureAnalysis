# Flash DSC Fictive Temperature Analysis and Step-Response Dynamics codebase


### To-do:
* Make as a package
* Seperate file reading, Tf, and dynamics code into seperate packages?
* Add functionality to choose baselines in Tf calculation. Currently the Tf is calculated per curve over T1 and T2. This is fine if the T1 and T2 are the same as for the baseline correction, then all curves have (by construction) the same baselines as the reference. But if, for example, you want to calculate the glass line from an aged curve but use the liquid line from the reference, this is not possible. Can be fixed by adding optional arguments to the Tf calculation that accept baseline objects.
* Clean up the analysis summary
* Finish and clean up documentation
* Fix git stuff
* Make clearer example notebooks
