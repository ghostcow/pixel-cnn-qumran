Instructions for running word spotting:

1. Extract the files from the zip file to a certain location. Keep everything as it is even the empty directories.
   You also need to download vlfeat and follow the instructions on-line according to your operating system. Add the vlfeat directory to the Word Spotting folder.

2. Copy all the documents to 'comp/Segmentation-free_Dataset/Documents'

3. The main function is 'run_comp'. When running the code for the first time make sure all the parameters are set properly in the beginning of the main function, and set the parameters in
   the configuration function 'get_default_cfg' according to your documents.

4. After running the first function 'read_pages' - the file 'P.mat' will be saved to 'comp/Segmentation-free_Dataset'.
   When running it again it would load 'P.mat' and not read the pages again. If you want it to run again delete 'P.mat'.

5. After running the function 'pre_process_pages2' - the file 'PPP.mat' will be saved to 'comp/Segmentation-free_Dataset'.
   The processed pages will be saved in 'comp/Segmentation-free_Dataset/BW' and in 'comp/Segmentation-free_Dataset/Pages'.
   When running it again it would load 'PPP.mat' and not process the pages again. If you want it to run again delete 'PPP.mat' and delete the files in 'BW' and 'Pages' folders.

6. After running 'build_windows_parfor' - all candidates will be computed and 'Segments.mat' will be saved to 'comp/Segmentation-based_Dataset'
   As before when running it again it would load 'Segments.mat' and not find the candidates again. If you want it to run again delete 'Segments.mat'.

7. The function 'create_model_build_matrix' is the function which creates the model matrix, with random candidates. The number of candidates is determined in 'get_default_cfg'. The function 'create_model_build_pages' computes the final candidates matrix. The size of each candidate vector is determined in 'get_default_cfg'.
   Make sure in the beginning of the main function you choose a location which have enough space to store 'Model.mat'.
   Again , a file 'Model.mat' will be saved in the specified location after running the second function.
   If you want to compute the model again and not load 'Model.mat' make sure you delete or move this file.

8. After 'Model.mat' has been saved, the building stage is over and it is possible to run queries.
   In 'Segmentation-free_Queries' put images of queries and create a file 'queries.gtp'.
   Each line in the text file should look like this: 'image_name text', for example - 'cand455.jpg Orders'
   When reading the queries a file 'Q.mat' is saved to 'Segmentation-free_GNIZA_Queries'. If you want to read the queries again or add more queries delete 'Q.mat'.

9. To get the results run 'run_queries2' and 'save_results2'.
	'Results.mat' will be saved to 'Segmentation-free_GNIZA_Queries'' so if you want to run queries again delete this file, otherwise it will load it.
As defualt when running queries there's a post processing stage, which is re-ranking the results using the full descriptors matrix of the candidates. If you want to disable this stage you can change the flag in 'get_default_cfg'.

     XML - is a cell array. The length of the array is the number of queries. each cell is a struct array containing all the results for one query.
         each struct in the array contains the following fields -
         doc - full path to the document which contains the current result
         x - x coordinate of the result
         y - y coordinate of the result
         w - width  of the result
         h - height of the result

10. Run 'save_images_results' to save the results images for each query in the 'results' folder, And run 'create_result_image' to create one image of the results in the same folder.



