Instructions for running word spotting:

1. Extract the files from the zip file to a certain location. Keep everything as it is even the empty directories.
   You also need to download vlfeat and follow the instructions on-line according to your operating system.
   Add the vlfeat directory to the letter_spotting folder.

2. Copy the document to 'comp/Segmentation-free_Dataset/Documents'

3. The main function is 'run_comp'. When running the code for the first time make sure all the parameters are set properly
   in the beginning of the main function, and set the parameters in the configuration function 'get_default_cfg'
   according to your documents.

4. After running the first function 'read_pages' - the file 'P.mat' will be saved to 'comp/Segmentation-free_Dataset'.
   When running it again it would load 'P.mat' and not read the pages again. If you want it to run again delete 'P.mat'.

5. After running the function 'pre_process_pages2' - the file 'PPP.mat' will be saved to 'comp/Segmentation-free_Dataset'.
   The processed pages will be saved in 'comp/Segmentation-free_Dataset/BW' and in 'comp/Segmentation-free_Dataset/Pages'.
   When running it again it would load 'PPP.mat' and not process the pages again. If you want it to run again
   delete 'PPP.mat' and delete the files in 'BW' and 'Pages' folders.

6. After running 'build_windows_parfor' - all candidates will be computed and 'Segments.mat' will be saved to
   'comp/Segmentation-based_Dataset' as before when running it again it would load 'Segments.mat' and not find
   the candidates again. If you want it to run again delete 'Segments.mat'.


   --------------

     XML - is a cell array. The length of the array is the number of queries. each cell is a struct array containing all the results for one query.
         each struct in the array contains the following fields -
         doc - full path to the document which contains the current result
         x - x coordinate of the result
         y - y coordinate of the result
         w - width  of the result
         h - height of the result