from matplotlib import pyplot as plt
import numpy as np

def result_plt(r_epochs, record_df):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    plt.show()

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()


    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()

    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    plt.show()

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')
    plt.show()

    plt.title('loss')
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'b')
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'g')
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'c')
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'm')
    plt.show()