
#!/bin/bash

# 训练模型
echo "开始训练模型..."
python train.py \
    --distributed \
    --world_size=4 \
    --epoch=40 \
    --batch_size=128 \
    --use_gpu \


# 测试模型
echo "开始测试模型..."
python test.py \
    --use_gpu \
    --beam_size=5

echo "训练和测试完成！"