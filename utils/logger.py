import logging
import os, datetime


def setup_logger(name, log_file, level=logging.INFO):
    """
    name: logger name, ex) 'first logger'
    log_file: save path
    Example---------------------------
    # first file logger
    logger = setup_logger('first_logger', 'first_logfile.log')
    logger.info('This is just info message')
    # second file logger
    super_logger = Setup_Logger('second_logger', 'second_logfile.log')
    super_logger.error('This is an error message')
    ----------------------------------
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def args_logger(args, path):
    """
    Print all arguments in argparser using a file operation
    Doesn't use the Python logger
    """
    path_checker(path)
    path = path + f"args_{args.save_model}.txt"

    with open(path, 'w') as f:
        for arg in vars(args):
            m = f"- {arg}: {getattr(args, arg)}\n"
            f.write(m)


def model_logger(path, networks):
    """
    Log a model
    """
    path = path + "network_log.txt"
    with open(path, 'w') as f:
        print(networks, file=f)
        f.write("\n-----------------------------------------------------")
        f.write("\n-----------------------------------------------------")


def path_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_path(now, path, save_model):
    path_checker(path)
    fname = f"{save_model}_{now.month}_{now.day}.log"
    return path + fname


if __name__ == "__main__":
    now = datetime.datetime.now()
    args.experiment_folder = args.experiment_folder + f"{now.year}_{now.month}_{now.day}/"
    print(f"Experiment Dir: {args.experiment_folder}")

    args_logger(args, args.experiment_folder)

    log_path = log_path(args.experiment_folder, args.n_window, train=False)
    performance_logger = logging_tool.setup_logger('test_info', log_path)
    # performance_logger.info(f"Epoch: {epoch} || Test accuracy result: {acc_epoch} || Best result so far: {Best_Result}||")

    # oov_logger = logging_tool.Setup_Logger('OOV_info', './oov_info.log')
