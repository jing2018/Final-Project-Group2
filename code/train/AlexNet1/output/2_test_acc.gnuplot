

reset
set terminal png
set output "test_acc.png"
set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set title "Test accuracy vs. training iterations"
set xlabel "Test accuracy"
set ylabel "training iterations"

###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test accuracy vs. training iterations
plot "Alex_450000_log-24042018_002909.log.test" using 1:3 title "Alexnet"



