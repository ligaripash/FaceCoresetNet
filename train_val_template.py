import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
from torch.nn import CrossEntropyLoss
import head
import net_template as net
import utils
from validation_IJBB_IJBC.validate_IJB_BC_compute_templates import validate_model_ijb



print('Tracing back tensors:')
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


class FaceCoresetNet(LightningModule):
    def __init__(self, **kwargs):
        super(FaceCoresetNet, self).__init__()
        #wandb.watch(self, 'all')
        self.save_hyperparameters()  # sets self.hparams

        #self.my_wandb = kwargs['logger']

        if self.hparams.train_data_path == 'faces_emore/imgs':
            class_num = 70722 if self.hparams.train_data_subset else 85742
        elif self.hparams.train_data_path == 'ms1m-retinaface-t1/imgs':
            assert not self.hparams.train_data_subset
            class_num = 93431
        elif self.hparams.train_data_path == 'WebFace4M':
            assert not self.hparams.train_data_subset
            class_num = 205990
        elif os.path.basename(self.hparams.train_data_path) == 'webface4m_subset_images':
            class_num = 10000
        else:
            raise ValueError('Check your train_data_path', self.hparams.train_data_path)

        self.class_num = class_num
        print('classnum: {}'.format(self.class_num))

        self.model = net.build_model(model_name=self.hparams.arch)


        self.head = head.build_head(head_type=self.hparams.head,
                                     embedding_size=512,
                                     class_num=class_num,
                                     m=self.hparams.m,
                                     h=self.hparams.h,
                                     t_alpha=self.hparams.t_alpha,
                                     s=self.hparams.s,
                                     )


        #if self.hparams.start_from_model_statedict and not self.hparams.resume_from_checkpoint:
        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            #self.load_state_dict(ckpt['state_dict'], strict=True)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                # from pytorch lightning checkpoint
                model_statedict = ckpt['state_dict']
                renamed = {key[6:]: val for key, val in model_statedict.items() if key.startswith('model.')}
                toupdate_statedict = self.model.state_dict()
                assert len(renamed) == len(toupdate_statedict)
                len_orig = len(toupdate_statedict)
                toupdate_statedict.update(renamed)
                assert len(toupdate_statedict) == len_orig
                match_res = self.model.load_state_dict(toupdate_statedict, strict=False)

        #This model accepts a tensor [N, T, C, H, W] and produce [N, T, 512] embeddings and [N, T, 1] norms
        self.template_model = net.TimeDistributed(self.model, batch_first=True)


        #This model aggregates [N, T, 512] template embeddings and produce a simgle aggregate per template
        #[N, 512]
        embedding_size = 512
        self.aggregate_model = net.TemplateAggregateModel(embedding_size, self.hparams.coreset_size, self.hparams.gamma)


        #Freeze weights of the backbone. We want to train only the aggregation module
        for param in self.template_model.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

        for param in self.aggregate_model.parameters():
            param.requires_grad = True


        self.cross_entropy_loss = CrossEntropyLoss()

        self.compute_feature_flag = False

        #self.gating = torch.nn.Linear(512, 512)


    def compute_feature(self):
        self.compute_feature_flag = True
    def forward(self, templates=None, labels=None, embeddings=None, norms=None):
        #embeddings, norms = self.model(images)
        #On inference mode we get the per template features as input:
        # (N, template_size, 512)
        #gil


        if self.training or self.compute_feature_flag:
            embeddings, norms = self.template_model(templates)
            if self.compute_feature_flag:
                unnorm_embeddings = embeddings * norms
                return unnorm_embeddings, embeddings

        # unnorm_embeddings = embeddings * norms
        # gating = self.gating(unnorm_embeddings).sigmoid()
        # unnorm_embeddings = unnorm_embeddings * gating
        # norms = unnorm_embeddings.norm(dim=-1).unsqueeze(-1)
        # embeddings = unnorm_embeddings / norms
        #norms = norms.squeeze(-1)
        aggregate_embeddings, aggregate_norms, FPS_sample = self.aggregate_model(embeddings, norms)


        if not self.training:
            # small_norms = aggregate_norms < 0
            # aggregate_embeddings[small_norms.squeeze()] = torch.zeros_like(aggregate_embeddings[small_norms.squeeze()])
            # aggregate_norms[small_norms.squeeze()] = 0
            return aggregate_embeddings, aggregate_norms, FPS_sample

        cos_thetas = self.head(aggregate_embeddings, aggregate_norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index

        #return cos_thetas, norms, embeddings, labels
        return cos_thetas, aggregate_norms, aggregate_embeddings, labels


    def training_step(self, batch, batch_idx):
        images, labels = batch
        #gil
        #self.eval()
        self.template_model.eval()
        cos_thetas, norms, embeddings, labels = self.forward(images, labels)
        loss_train = self.cross_entropy_loss(cos_thetas, labels)
        #lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]

        lr_gamma = self.optimizers().optimizer.param_groups[0]['lr']
        lr_other = self.optimizers().optimizer.param_groups[1]['lr']

        # log
        log_interval = 10
        # with torch.no_grad():
        #     dist_ur_learned_empirical = (self.aggregate_model.ur_center_empirical.detach().cpu() - self.aggregate_model.ur_center.detach().cpu()).norm()

        # if self.local_rank == 0:
        #     if batch_idx % log_interval == 0:
        #         self.log({'lr': lr,
        #                    'train_loss': loss_train,
        #                    'gamma': self.aggregate_model.gamma.float().detach().cpu().numpy(),
        #                    'dist_ur_center': dist_ur_learned_empirical.detach().cpu().numpy(),
        #                    'dist_from_ur_center_norm_size_balance':self.aggregate_model.dist_from_ur_center_norm_size_balance.detach().cpu().numpy()})

        self.log_dict({'lr_gamma': lr_gamma,
                       'lr_other': lr_other,
                   'train_loss': loss_train,
                   'gamma': self.aggregate_model.gamma.float().detach().cpu(),
                       'tau': self.aggregate_model.tau.sigmoid().float().detach().cpu(),
                       'h': self.hparams.h})




        return loss_train

    def on_after_backward(self):
        self.log_dict({'gamma_gradint_norm': self.aggregate_model.gamma.grad.data.norm(2)})


    def on_train_epoch_end(self):
        if self.local_rank == 0:
            self.eval()
            output_dir = '{}/epoch_{}'.format(self.hparams.output_dir, self.current_epoch)
            print('gamma = {}'.format(self.aggregate_model.gamma))
            scores_ijbb = validate_model_ijb(self, output_dir, self.hparams.ijb_root, dataset_name='IJBB', args=self.hparams)
            scores_ijbc = validate_model_ijb(self, output_dir, self.hparams.ijb_root, dataset_name='IJBC', args=self.hparams)



            self.log_dict({'epoch':self.current_epoch,
                       '10 ** -6':scores_ijbb[0],
                       '10 ** -5':scores_ijbb[1],
                       '10 ** -4':scores_ijbb[2],
                       '10 ** -3':scores_ijbb[3],
                       '10 ** -2':scores_ijbb[4],
                       '10 ** -1':scores_ijbb[5],
                           '10 ** -6 IJBC': scores_ijbc[0],
                           '10 ** -5 IJBC': scores_ijbc[1],
                           '10 ** -4 IJBC': scores_ijbc[2],
                           '10 ** -3 IJBC': scores_ijbc[3],
                           '10 ** -2 IJBC': scores_ijbc[4],
                           '10 ** -1 IJBC': scores_ijbc[5],

                           })
            self.train()
        return None

    # def validation_step(self, batch, batch_idx):
    #     return None

    def test_step(self, batch, batch_idx):
        #return self.validation_step(batch, batch_idx)
        return None


    def test_epoch_end(self, outputs):
        return None

        #if self.local_rank == 0:
        self.eval()
        output_dir = '{}/epoch_{}'.format(self.hparams.output_dir, self.current_epoch)
        print('gamma = {}'.format(self.aggregate_model.gamma))
        scores = validate_model_ijb(self, output_dir, self.hparams.ijb_root)
        total_scores = sum(scores)

        # create a wandb.Table() with corresponding columns
        #table_row = [self.current_epoch] + scores
        #self.roc_table.add_data(*table_row)
        ['epoch', '10 ** -6', '10 ** -5', '10 ** -4', '10 ** -3', '10 ** -2', '10 ** -1']
        self.log_dict({'total_6_point_roc': total_scores,
                   'epoch':self.current_epoch,
                   '10 ** -6':scores[0],
                   '10 ** -5':scores[1],
                   '10 ** -4':scores[2],
                   '10 ** -3':scores[3],
                   '10 ** -2':scores[4],
                   '10 ** -1':scores[5]})

        return None


    def gather_outputs(self, outputs):
        if self.hparams.distributed_backend == 'ddp':
            # gather outputs across gpu
            outputs_list = []
            _outputs_list = utils.all_gather(outputs)
            for _outputs in _outputs_list:
                outputs_list.extend(_outputs)
        else:
            outputs_list = outputs

        # if self.trainer.is_global_zero:
        all_output_tensor = torch.cat([out['output'] for out in outputs_list], axis=0).to('cpu')
        all_norm_tensor = torch.cat([out['norm'] for out in outputs_list], axis=0).to('cpu')
        all_target_tensor = torch.cat([out['target'] for out in outputs_list], axis=0).to('cpu')
        all_dataname_tensor = torch.cat([out['dataname'] for out in outputs_list], axis=0).to('cpu')
        all_image_index = torch.cat([out['image_index'] for out in outputs_list], axis=0).to('cpu')

        # get rid of duplicate index outputs
        unique_dict = {}
        for _out, _nor, _tar, _dat, _idx in zip(all_output_tensor, all_norm_tensor, all_target_tensor,
                                                all_dataname_tensor, all_image_index):
            unique_dict[_idx.item()] = {'output': _out, 'norm': _nor, 'target': _tar, 'dataname': _dat}
        unique_keys = sorted(unique_dict.keys())
        all_output_tensor = torch.stack([unique_dict[key]['output'] for key in unique_keys], axis=0)
        all_norm_tensor = torch.stack([unique_dict[key]['norm'] for key in unique_keys], axis=0)
        all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
        all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)

        return all_output_tensor, all_norm_tensor, all_target_tensor, all_dataname_tensor

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        metric_params = [self.aggregate_model.gamma]
        #non_metric_params = [ *self.aggregate_model.encoder_layer.parameters(), *self.aggregate_model.decoder_layer1.parameters(), *self.aggregate_model.decoder_layer2.parameters(), *self.head.parameters()]
        all_params = [*self.aggregate_model.parameters(),
                             *self.head.parameters()]

        s_all_params = set(all_params)
        s_metric_params = set(metric_params)
        non_metric_params = s_all_params.difference(s_metric_params)
        non_metric_params = list(non_metric_params)

        #all_other_params = [item for item in self.parameters() if item not in metric_params]

        optimizer = torch.optim.AdamW([{"params": metric_params, "lr": self.hparams.gamma_lr, "weight_decay":0.0},
                                       {"params": non_metric_params, "lr": self.hparams.lr, "weight_decay":self.hparams.weight_decay}])

        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=self.hparams.lr_milestones,
                                                 gamma=self.hparams.lr_gamma)

        grad_hook1 = self.aggregate_model.gamma.register_hook(lambda grad: print("Grad is {0}".format(grad)))


        return [optimizer], [scheduler]

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay
