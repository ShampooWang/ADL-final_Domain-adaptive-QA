from datasets import load_dataset, concatenate_datasets, interleave_datasets


def get_train_dataset(seed, args):
    dataset_names = args.train_dataset_names.split()

    train_dataset = []
    dataset_num = {}
    for tag in dataset_names:
        dataset = load_dataset('loaders/eqa_loader.py', name=tag)['train']
        args_num = args.max_train_samples
        if args_num is not None:
            dataset = dataset.shuffle(seed)
            if args_num > len(dataset):
                args_num = len(dataset)
                print(f"args.max_train_samples {args.max_train_samples} > dataset_num {len(dataset)}, so we use whole dataset_num {args_num}")
            dataset = dataset.select(range(args_num))
        dataset_num[tag] = len(dataset)
        train_dataset.append(dataset)
    print(f"\ntrain_dataset_num: {dataset_num}\n")
    if args.interleave:
        train_dataset = interleave_datasets(train_dataset)
    else:
        train_dataset = concatenate_datasets(train_dataset)

    return train_dataset


def get_eval_dataset(seed, args):
    dataset_names = args.eval_dataset_names.split()

    eval_dataset = {}
    dataset_num = {}
    for tag in dataset_names:
        dataset = load_dataset('loaders/eqa_loader.py', name=tag)['validation']
        args_num = args.max_eval_samples
        if args_num is not None:
            dataset = dataset.shuffle(seed)
            if args_num > len(dataset):
                args_num = len(dataset)
                print(f"args.max_eval_samples {args.max_eval_samples} > dataset_num {len(dataset)}, so we use whole dataset_num {args_num}")
            dataset = dataset.select(range(args_num))
        dataset_num[tag] = len(dataset)
        eval_dataset.update({tag: dataset})
    print(f"\neval_dataset_num: {dataset_num}\n")
    return eval_dataset


def get_train_mrqa(seed, args):
    dataset = load_dataset('mrqa')

    print(dataset)
    print(dataset['train'][0])

    return dataset['train']


if __name__ == '__main__':
    get_train_mrqa()

