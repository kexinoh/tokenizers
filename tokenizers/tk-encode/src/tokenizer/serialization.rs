use std::marker::PhantomData;

use serde::{
    self, Deserialize, Deserializer, Serialize, Serializer,
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
};

use super::TokenizerImpl;
use super::added_vocabulary::AddedTokenWithId;
use crate::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerBuilder};

static SERIALIZATION_VERSION: &str = "1.0";

impl<M, N, PT, PP, D> Serialize for TokenizerImpl<M, N, PT, PP, D>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
    PP: Serialize,
    D: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 10)?;

        // Start by adding the current version
        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;

        // Params
        tokenizer.serialize_field("truncation", &self.truncation)?;
        tokenizer.serialize_field("padding", &self.padding)?;

        // Which token plays which role, so this file can stand in for `tokenizer_config.json`
        tokenizer.serialize_field("role_to_token", &self.role_to_token)?;

        // Added tokens
        tokenizer.serialize_field("added_tokens", &self.added_vocabulary)?;

        // Then add our parts
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("post_processor", &self.post_processor)?;
        tokenizer.serialize_field("decoder", &self.decoder)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de, M, N, PT, PP, D> Deserialize<'de> for TokenizerImpl<M, N, PT, PP, D>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tokenizer",
            &[
                "version",
                "truncation",
                "padding",
                "role_to_token",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            ],
            TokenizerVisitor(
                PhantomData,
                PhantomData,
                PhantomData,
                PhantomData,
                PhantomData,
            ),
        )
    }
}

struct TokenizerVisitor<M, N, PT, PP, D>(
    PhantomData<M>,
    PhantomData<N>,
    PhantomData<PT>,
    PhantomData<PP>,
    PhantomData<D>,
);

impl<'de, M, N, PT, PP, D> Visitor<'de> for TokenizerVisitor<M, N, PT, PP, D>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
{
    type Value = TokenizerImpl<M, N, PT, PP, D>;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Tokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = TokenizerBuilder::new();
        let mut tokens: Vec<AddedTokenWithId> = vec![];
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "version" => {
                    let v: String = map.next_value()?;
                    if &v != "1.0" {
                        return Err(Error::custom(format!("Unknown tokenizer version '{v}'")));
                    }
                }
                "truncation" => {
                    builder = builder.with_truncation(map.next_value()?);
                }
                "padding" => {
                    builder = builder.with_padding(map.next_value()?);
                }
                "role_to_token" => {
                    builder = builder.with_role_to_token(map.next_value()?);
                }
                "added_tokens" => {
                    tokens = map.next_value()?;
                }
                "normalizer" => {
                    builder = builder.with_normalizer(map.next_value()?);
                }
                "pre_tokenizer" => {
                    builder = builder.with_pre_tokenizer(map.next_value()?);
                }
                "model" => {
                    builder = builder.with_model(map.next_value()?);
                }
                "decoder" => {
                    builder = builder.with_decoder(map.next_value()?);
                }
                "post_processor" => {
                    builder = builder.with_post_processor(map.next_value()?);
                }
                _ => {}
            };
        }
        let mut tokenizer = builder
            .build()
            .map_err(|e| V::Error::custom(e.to_string()))?;

        // Single-pass: warn on ID mismatches, then add all tokens.
        // `add_tokens` computes normalization internally for tokens with `normalized = true`.
        for t in &tokens {
            if let Some(rid) = tokenizer.token_to_id(&t.token.content)
                && rid != t.id
            {
                warn!(
                    "Warning: Token '{}' was expected to have ID '{}' but was given ID '{}'",
                    t.token.content, t.id, rid
                );
            }
        }
        tokenizer
            .add_tokens(tokens.into_iter().map(|t| t.token))
            .map_err(|e| V::Error::custom(e.to_string()))?;

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Tokenizer;
    use std::str::FromStr;

    #[test]
    fn test_deserialization_serialization_invariant() {
        let tok_json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "role_to_token": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "[SPECIAL_0]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 1,
      "content": "[SPECIAL_1]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 2,
      "content": "[SPECIAL_2]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "vocab": {}
  }
}"#;
        let tokenizer = Tokenizer::from_str(tok_json).unwrap();

        let tok_str = serde_json::to_string_pretty(&tokenizer).unwrap();
        // It should be exactly the same as above
        assert_eq!(tok_str, tok_json);
    }

    #[test]
    fn role_to_token_survives_a_round_trip() {
        use crate::models::wordpiece::WordPiece;
        use crate::tokenizer::AddedToken;
        use std::collections::HashMap;

        let mut tokenizer = Tokenizer::new(WordPiece::default());
        tokenizer
            .add_special_tokens([
                AddedToken::from("</s>", true),
                AddedToken::from("<s>", true),
                AddedToken::from("<pad>", true),
            ])
            .unwrap();

        let mut roles = HashMap::new();
        roles.insert("eos_token".to_string(), "</s>".to_string());
        roles.insert("bos_token".to_string(), "<s>".to_string());
        roles.insert("pad_token".to_string(), "<pad>".to_string());
        tokenizer.with_role_to_token(Some(roles));

        let ser = serde_json::to_string(&tokenizer).unwrap();
        assert!(ser.contains("role_to_token"));

        let de = Tokenizer::from_str(&ser).unwrap();
        let round_tripped = de
            .get_role_to_token()
            .expect("role_to_token should survive");
        assert_eq!(round_tripped.get("eos_token"), Some(&"</s>".to_string()));
        assert_eq!(round_tripped.get("bos_token"), Some(&"<s>".to_string()));
        assert_eq!(round_tripped.get("pad_token"), Some(&"<pad>".to_string()));
    }

    #[test]
    fn role_to_token_resolves_tokens_and_ids() {
        use crate::models::wordpiece::WordPiece;
        use crate::tokenizer::AddedToken;
        use std::collections::HashMap;

        let mut tokenizer = Tokenizer::new(WordPiece::default());
        tokenizer
            .add_special_tokens([
                AddedToken::from("</s>", true),
                AddedToken::from("<unk>", true),
            ])
            .unwrap();

        let mut roles = HashMap::new();
        roles.insert("eos_token".to_string(), "</s>".to_string());
        roles.insert("unk_token".to_string(), "<unk>".to_string());
        tokenizer.with_role_to_token(Some(roles));

        assert_eq!(
            tokenizer.get_token_for_role("eos_token"),
            Some(&"</s>".to_string())
        );
        assert_eq!(tokenizer.get_token_for_role("nonexistent"), None);

        // Resolved through the vocab, in the order the tokens were added.
        assert_eq!(tokenizer.get_id_for_role("eos_token"), Some(0));
        assert_eq!(tokenizer.get_id_for_role("unk_token"), Some(1));
        assert_eq!(tokenizer.get_id_for_role("nonexistent"), None);
    }

    /// An untouched tokenizer still reports no roles, and absent `role_to_token` deserializes fine.
    #[test]
    fn role_to_token_defaults_to_none() {
        let tok_json = r#"{"version":"1.0","model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"","max_input_chars_per_word":100,"vocab":{}}}"#;
        let tokenizer = Tokenizer::from_str(tok_json).unwrap();
        assert!(tokenizer.get_role_to_token().is_none());
        assert_eq!(tokenizer.get_token_for_role("eos_token"), None);
        assert_eq!(tokenizer.get_id_for_role("eos_token"), None);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_pretrained() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_target(false)
            .init();
        let _ = Tokenizer::from_pretrained("Qwen/Qwen2-7B-Instruct", None);
        warn!("This should be the first warning");
    }
}
