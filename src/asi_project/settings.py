from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional `.env`."""

    MODEL_PATH: str | None = None
    WANDB_API_KEY: str | None = None
    DATABASE_URL: str = "sqlite:///local.db"

    MODEL_VERSION: str | None = None
    WANDB_PROJECT: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
