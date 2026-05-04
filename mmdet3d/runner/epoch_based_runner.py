import shutil
import os.path as osp

from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset

    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch)
        super().train(data_loader, **kwargs)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth',
                        save_optimizer=True, meta=None, create_symlink=True):
        """Override to replace symlink with file copy for NFS/unsupported filesystems."""
        super().save_checkpoint(
            out_dir,
            filename_tmpl=filename_tmpl,
            save_optimizer=save_optimizer,
            meta=meta,
            create_symlink=False,   # disable symlink, use copy below
        )
        # mmcv saves epoch_{self.epoch+1}.pth, so use same formula
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        dst_file = osp.join(out_dir, 'latest.pth')
        try:
            shutil.copyfile(filepath, dst_file)
        except Exception as e:
            self.logger.warning(
                f'[CustomEpochBasedRunner] Failed to copy latest.pth: {e}'
            )
