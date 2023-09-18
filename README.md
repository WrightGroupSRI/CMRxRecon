# Sunnybrook CMRxRecon Submission

To prepare the code, modify the file `docker-test.sh` to point to the correct path to the validation data and the output predictions. To use the training or testing set, modify `code/main.py` to look for the training or testing data. Additionally, for the testing set, to place the data into the required final shape, modify `code/main.py` to include a final transpose step (currently commented out). To obtain the model weights, please download the file `weights.pt` from this link:  https://drive.google.com/file/d/1zWnHqrxsCwLI8hUslVSN4h1NN4c0Xf1r/view?usp=drive_link , and place it in `./code/`

To run the code, execute the command `sh docker-test.sh`, and the container will be built and run with the specified file paths.



