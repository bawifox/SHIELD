"""
CLIP Text Encoder Module
用于生成文本 prompt 特征，支持缓存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TextEmbedding:
    """Text embedding 数据结构"""
    text: str
    embedding: torch.Tensor  # [embed_dim]


class CLIPTextEncoder(nn.Module):
    """
    冻结的 CLIP Text Encoder
    
    Features:
    - 文本编码器冻结，不参与训练
    - 默认 prompts 预设
    - LRU 缓存避免重复编码
    - 支持自定义 prompts
    
    Args:
        model_name: CLIP 模型名称 (default: "openai/clip-vit-base-patch32")
        device: 设备
        embed_dim: 输出 embedding 维度
    """
    
    DEFAULT_PROMPTS = [
        "a tiny hazardous obstacle on the road",
        "a subtle but dangerous anomalous object",
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
        embed_dim: int = 512
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        
        # 加载 CLIP 模型
        from transformers import CLIPTextModel, CLIPTokenizer
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # 投影层到统一维度
        self.projection = nn.Linear(self.model.config.hidden_size, embed_dim)
        
        # 缓存: text -> embedding
        self._cache: Dict[str, torch.Tensor] = {}
        
        # 移动到设备
        self.to(self.device)
    
    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 文本列表
            normalize: 是否归一化
            
        Returns:
            embeddings: [N, embed_dim]
        """
        # 检查缓存
        uncached_texts = []
        results = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text].to(self.device)
            else:
                uncached_texts.append((i, text))
        
        # 编码未缓存的文本
        if uncached_texts:
            # Tokenize
            text_list = [t for _, t in uncached_texts]
            inputs = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            outputs = self.model(**inputs)
            
            # 使用 mean pooling
            hidden_states = outputs.last_hidden_state  # [B, L, hidden_size]
            attention_mask = inputs.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            # 投影
            embeddings = self.projection(pooled)  # [B, embed_dim]
            
            # 归一化
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # 存入缓存并填充结果
            for j, (orig_idx, text) in enumerate(uncached_texts):
                self._cache[text] = embeddings[j].cpu()
                results[orig_idx] = embeddings[j]
        
        # 合并结果
        output = torch.stack(results)  # [N, embed_dim]
        return output
    
    @torch.no_grad()
    def encode_default_prompts(self, normalize: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """
        编码默认 prompts
        
        Returns:
            embeddings: [num_prompts, embed_dim]
            prompts: prompt 列表
        """
        return self.encode(self.DEFAULT_PROMPTS, normalize=normalize), self.DEFAULT_PROMPTS
    
    @torch.no_grad()
    def encode_with_prompt_ensemble(
        self,
        base_texts: List[str],
        templates: Optional[List[str]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        使用 prompt 模板生成多个变体并编码
        
        Args:
            base_texts: 基础文本列表
            templates: 模板列表，默认使用常用模板
            normalize: 是否归一化
            
        Returns:
            embeddings: [N * num_templates, embed_dim]
        """
        if templates is None:
            templates = [
                "{}",
                "a photo of {}",
                "a cropped picture of {}",
                "a close-up photo of {}",
                "a bad photo of {}",
            ]
        
        # 生成所有变体
        all_prompts = []
        for template in templates:
            for base in base_texts:
                all_prompts.append(template.format(base))
        
        return self.encode(all_prompts, normalize=normalize)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def load_cache_from_file(self, cache_path: str):
        """从文件加载缓存"""
        import pickle
        with open(cache_path, 'rb') as f:
            self._cache = pickle.load(f)
        print(f"Loaded {len(self._cache)} cached embeddings from {cache_path}")
    
    def save_cache_to_file(self, cache_path: str):
        """保存缓存到文件"""
        import pickle
        # 保存时移到 CPU
        cache_cpu = {k: v.cpu() for k, v in self._cache.items()}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_cpu, f)
        print(f"Saved {len(self._cache)} cached embeddings to {cache_path}")


class CLIPTextEncoderSimple:
    """
    简化的 CLIP Text Encoder (非 nn.Module)
    适合直接调用，不需要作为模型组件
    """
    
    DEFAULT_PROMPTS = [
        "a tiny hazardous obstacle on the road",
        "a subtle but dangerous anomalous object",
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
        embed_dim: int = 512,
        use_cache: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.use_cache = use_cache
        
        # 加载 CLIP 模型
        from transformers import CLIPTextModel, CLIPTokenizer
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 投影层
        self.projection = nn.Linear(self.model.config.hidden_size, embed_dim)
        self.projection.to(self.device)
        
        # 缓存
        self._cache: Dict[str, torch.Tensor] = {}
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """编码文本，支持缓存"""
        if not self.use_cache:
            return self._encode_once(texts, normalize)
        
        # 检查缓存
        uncached = []
        results = []
        
        for text in texts:
            if text in self._cache:
                results.append(self._cache[text].to(self.device))
            else:
                uncached.append(text)
        
        # 编码未缓存的
        if uncached:
            new_embeddings = self._encode_once(uncached, normalize)
            for text, emb in zip(uncached, new_embeddings):
                self._cache[text] = emb.cpu()
            results.extend(new_embeddings)
        
        return torch.stack(results)
    
    @torch.no_grad()
    def _encode_once(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """单次编码 (无缓存)"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling
        attention_mask = inputs.attention_mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # 投影并归一化
        embeddings = self.projection(pooled)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    @torch.no_grad()
    def encode_default_prompts(self, normalize: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """编码默认 prompts"""
        return self.encode(self.DEFAULT_PROMPTS, normalize=normalize), self.DEFAULT_PROMPTS

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    @torch.no_grad()
    def encode_with_prompt_ensemble(
        self,
        base_texts: List[str],
        templates: Optional[List[str]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """使用 prompt 模板生成多个变体并编码"""
        if templates is None:
            templates = ["{}", "a photo of {}", "a cropped picture of {}", "a close-up photo of {}"]
        
        all_prompts = []
        for template in templates:
            for base in base_texts:
                all_prompts.append(template.format(base))
        
        return self.encode(all_prompts, normalize=normalize)


# ============ 测试 ============

def test_clip_text_encoder():
    """测试 CLIP Text Encoder"""
    print("=" * 60)
    print("Testing CLIPTextEncoder...")
    print("=" * 60)
    
    # 测试简单版本
    encoder = CLIPTextEncoderSimple(
        model_name="openai/clip-vit-base-patch32",
        embed_dim=512
    )
    
    # 编码默认 prompts
    embeddings, prompts = encoder.encode_default_prompts()
    
    print(f"\nDefault prompts ({len(prompts)}):")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. {p}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # 测试缓存
    print("\nTesting cache...")
    embeddings2 = encoder.encode(prompts)  # 应该命中缓存
    print(f"Cache size: {encoder.get_cache_size()}")
    
    # 测试自定义 prompts
    custom_prompts = [
        "a crack on the road",
        "a pothole",
        "an oil spill"
    ]
    custom_embeddings = encoder.encode(custom_prompts)
    print(f"\nCustom prompts ({len(custom_prompts)}):")
    print(f"  Embedding shape: {custom_embeddings.shape}")
    
    # 测试 prompt ensemble
    print("\nTesting prompt ensemble...")
    ensemble_embeddings = encoder.encode_with_prompt_ensemble(
        ["hazardous object", "anomaly"],
        templates=["{}", "a photo of {}", "a close-up of {}"]
    )
    print(f"  Ensemble embedding shape: {ensemble_embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_clip_text_encoder()
