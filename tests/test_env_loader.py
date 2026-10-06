"""
配置加载器测试
----------------
重点验证两件事：

1. **优先级正确**：项目专属变量必须压过通用兼容变量，
   否则改名就失去意义（外部工具改一次系统环境变量就能污染本项目）。
2. **降级不崩**：.env 缺失、内容异常时不能让整个框架起不来。

这些用例全部离线，不调用任何 API。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from eval import env_loader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# .env 文本解析
# ============================================================
class TestParseEnvText:

    def test_basic_pairs(self):
        d = env_loader.parse_env_text("A=1\nB=2")
        assert d == {"A": "1", "B": "2"}

    def test_skips_comments_and_blanks(self):
        text = "# 注释\n\nA=1\n   \n# 另一行注释\nB=2"
        assert env_loader.parse_env_text(text) == {"A": "1", "B": "2"}

    def test_strips_double_quotes(self):
        d = env_loader.parse_env_text('KEY="https://example.com/v1"')
        assert d["KEY"] == "https://example.com/v1"

    def test_strips_single_quotes(self):
        assert env_loader.parse_env_text("KEY='abc'")["KEY"] == "abc"

    def test_hash_inside_quotes_is_kept(self):
        """引号内的 # 不是注释，密钥里可能出现类似内容"""
        d = env_loader.parse_env_text('KEY="abc#def"')
        assert d["KEY"] == "abc#def"

    def test_trailing_comment_removed_when_unquoted(self):
        d = env_loader.parse_env_text("MODEL=glm-4-flash  # 默认模型")
        assert d["MODEL"] == "glm-4-flash"

    def test_export_prefix_supported(self):
        assert env_loader.parse_env_text("export KEY=v")["KEY"] == "v"

    def test_line_without_equals_ignored(self):
        d = env_loader.parse_env_text("JUST_A_LINE\nA=1")
        assert d == {"A": "1"}

    def test_empty_key_ignored(self):
        d = env_loader.parse_env_text("=novalue\nA=1")
        assert d == {"A": "1"}

    def test_crlf_line_endings(self):
        """Windows 记事本会存 CRLF，必须能解析"""
        d = env_loader.parse_env_text("A=1\r\nB=2\r\n")
        assert d == {"A": "1", "B": "2"}

    def test_later_key_overrides_earlier(self):
        assert env_loader.parse_env_text("A=1\nA=2")["A"] == "2"

    def test_value_may_contain_equals(self):
        """base64 之类的值里带 = 号不应被截断"""
        d = env_loader.parse_env_text("KEY=abc==def")
        assert d["KEY"] == "abc==def"

    def test_empty_value_is_kept_as_empty_string(self):
        d = env_loader.parse_env_text("EMPTY=")
        assert d["EMPTY"] == ""

    def test_empty_text(self):
        assert env_loader.parse_env_text("") == {}


# ============================================================
# 优先级
# ============================================================
class TestPriority:
    """
    隔离的核心保障。monkeypatch 隔离各测试间的环境变量。
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        # 清掉进程里可能存在的真实变量，保证测试可重复
        for name in (env_loader.ENV_API_KEY, env_loader.ENV_BASE_URL,
                     env_loader.ENV_MODEL_NAME, env_loader.ENV_JUDGE_MODEL,
                     env_loader.LEGACY_API_KEY, env_loader.LEGACY_BASE_URL,
                     env_loader.LEGACY_MODEL_NAME, env_loader.LEGACY_JUDGE_MODEL):
            monkeypatch.delenv(name, raising=False)
        # 把 .env 指向一个空文件，确保测试不受真实 .env 影响
        monkeypatch.setattr(env_loader, "ENV_FILE", str(tmp_path / ".env"))

    def test_project_name_in_env_wins_over_legacy_in_env(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "project-key")
        monkeypatch.setenv(env_loader.LEGACY_API_KEY, "legacy-key")
        assert env_loader.get_api_key() == "project-key"

    def test_project_in_envfile_wins_over_legacy_in_env(self, monkeypatch, tmp_path):
        """关键回归：外部工具改了系统环境变量时，.env 里的项目配置必须仍然赢。

        这正是引入 EVAL_ 前缀的意义所在。如果这条反过来，
        变量改名就等于白改。
        """
        env_file = tmp_path / ".env"
        env_file.write_text(f"{env_loader.ENV_API_KEY}=project-key\n",
                            encoding="utf-8")
        monkeypatch.setenv(env_loader.LEGACY_API_KEY, "legacy-key")
        assert env_loader.get_api_key() == "project-key"

    def test_project_in_env_wins_over_legacy_in_envfile(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            f"{env_loader.ENV_API_KEY}=project-key\n"
            f"{env_loader.LEGACY_API_KEY}=legacy-key\n",
            encoding="utf-8")
        assert env_loader.get_api_key() == "project-key"

    def test_fallback_to_legacy_when_project_unset(self, monkeypatch):
        """兼容性保障：老用户只配了 OPENAI_* 也必须能用"""
        monkeypatch.setenv(env_loader.LEGACY_API_KEY, "legacy-key")
        assert env_loader.get_api_key() == "legacy-key"

    def test_fallback_to_legacy_in_envfile(self, tmp_path):
        (tmp_path / ".env").write_text(
            f"{env_loader.LEGACY_API_KEY}=legacy-key\n", encoding="utf-8")
        assert env_loader.get_api_key() == "legacy-key"

    def test_default_used_when_nothing_set(self):
        assert env_loader.get("NOT_SET_AT_ALL", "ALSO_NOT_SET", "fallback") == "fallback"

    def test_empty_string_does_not_shadow_lower_priority(self, tmp_path):
        """空值等于未配置，应该继续往下找而不是直接返回空串

        否则 .env 模板里预留的空行会让整个配置看起来'已设置但为空'，
        排查起来非常困惑。
        """
        (tmp_path / ".env").write_text(f"{env_loader.ENV_API_KEY}=\n", encoding="utf-8")
        assert env_loader.get_api_key() is None


# ============================================================
# 具体配置项
# ============================================================
class TestAccessors:

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        for name in (env_loader.ENV_API_KEY, env_loader.ENV_BASE_URL,
                     env_loader.ENV_MODEL_NAME, env_loader.ENV_JUDGE_MODEL,
                     env_loader.LEGACY_API_KEY, env_loader.LEGACY_BASE_URL,
                     env_loader.LEGACY_MODEL_NAME, env_loader.LEGACY_JUDGE_MODEL):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(env_loader, "ENV_FILE", str(tmp_path / ".env"))

    def test_base_url_has_default(self):
        assert env_loader.get_base_url() == env_loader.DEFAULT_BASE_URL

    def test_model_name_has_default(self):
        assert env_loader.get_model_name() == env_loader.DEFAULT_MODEL_NAME

    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_BASE_URL, "https://api.deepseek.com/v1")
        assert env_loader.get_base_url() == "https://api.deepseek.com/v1"

    def test_judge_falls_back_to_sut_model(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_MODEL_NAME, "glm-4-flash")
        assert env_loader.get_judge_model_name() == "glm-4-flash"

    def test_judge_dedicated_wins(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_MODEL_NAME, "glm-4-flash")
        monkeypatch.setenv(env_loader.ENV_JUDGE_MODEL, "glm-4-plus")
        assert env_loader.get_judge_model_name() == "glm-4-plus"

    def test_api_key_env_name_reports_project_name(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "k")
        assert env_loader.api_key_env_name() == env_loader.ENV_API_KEY

    def test_api_key_env_name_reports_legacy_name(self, monkeypatch):
        monkeypatch.setenv(env_loader.LEGACY_API_KEY, "k")
        assert env_loader.api_key_env_name() == env_loader.LEGACY_API_KEY


# ============================================================
# 缺失校验
# ============================================================
class TestMissingRequired:

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        for name in (env_loader.ENV_API_KEY, env_loader.ENV_BASE_URL,
                     env_loader.ENV_MODEL_NAME, env_loader.ENV_JUDGE_MODEL,
                     env_loader.LEGACY_API_KEY, env_loader.LEGACY_BASE_URL,
                     env_loader.LEGACY_MODEL_NAME, env_loader.LEGACY_JUDGE_MODEL):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(env_loader, "ENV_FILE", str(tmp_path / ".env"))

    def test_nothing_configured(self):
        missing = [n for n, _ in env_loader.missing_required()]
        assert missing == [env_loader.ENV_API_KEY]

    def test_base_url_and_model_not_required(self):
        """这两个有默认值，缺失不算配置错误，否则新手会被无谓地拦住"""
        assert [n for n, _ in env_loader.missing_required()] == [env_loader.ENV_API_KEY]

    def test_all_configured(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "k")
        monkeypatch.setenv(env_loader.ENV_BASE_URL, "https://x")
        monkeypatch.setenv(env_loader.ENV_MODEL_NAME, "m")
        assert env_loader.missing_required() == []

    def test_legacy_names_satisfy_requirement(self, monkeypatch):
        monkeypatch.setenv(env_loader.LEGACY_API_KEY, "k")
        assert env_loader.missing_required() == []

    def test_check_env_raises_when_missing(self):
        with pytest.raises(EnvironmentError) as exc:
            env_loader.check_env()
        msg = str(exc.value)
        assert env_loader.ENV_API_KEY in msg
        # 报错信息必须给出可复制的操作路径
        assert ".env" in msg

    def test_check_env_passes_when_configured(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "k")
        assert env_loader.check_env() is True


# ============================================================
# 文件加载降级
# ============================================================
class TestLoadEnvFileDegradation:

    def test_missing_file_returns_empty(self, tmp_path):
        assert env_loader.load_env_file(str(tmp_path / "nope.env")) == {}

    def test_directory_instead_of_file_returns_empty(self, tmp_path):
        """路径存在但是个目录时不能抛异常，降级为空"""
        assert env_loader.load_env_file(str(tmp_path)) == {}

    def test_undecodable_file_returns_empty(self, tmp_path):
        """二进制垃圾文件不应让框架崩溃"""
        bad = tmp_path / "bad.env"
        bad.write_bytes(b"\xff\xfe\x00\x01binary garbage")
        assert env_loader.load_env_file(str(bad)) == {}

    def test_project_env_example_parses(self):
        """仓库里的模板文件必须始终可解析，防止文档腐化"""
        example = os.path.join(PROJECT_ROOT, "env.example")
        assert os.path.exists(example), "env.example 不存在"

        values = env_loader.load_env_file(example)
        assert env_loader.ENV_API_KEY in values, "模板缺少 EVAL_API_KEY"
        assert env_loader.ENV_BASE_URL in values
        assert env_loader.ENV_MODEL_NAME in values


# ============================================================
# 脱敏输出
# ============================================================
class TestDescribe:

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        for name in (env_loader.ENV_API_KEY, env_loader.ENV_BASE_URL,
                     env_loader.ENV_MODEL_NAME, env_loader.ENV_JUDGE_MODEL,
                     env_loader.LEGACY_API_KEY, env_loader.LEGACY_BASE_URL,
                     env_loader.LEGACY_MODEL_NAME, env_loader.LEGACY_JUDGE_MODEL):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(env_loader, "ENV_FILE", str(tmp_path / ".env"))

    def test_key_is_masked(self, monkeypatch):
        real = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
        monkeypatch.setenv(env_loader.ENV_API_KEY, real)
        shown = env_loader.describe()["api_key"]
        assert real not in shown, "完整密钥泄露到了输出里"
        assert shown.startswith("sk-a") and shown.endswith("89")

    def test_short_key_fully_masked(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "short")
        assert env_loader.describe()["api_key"] == "***"

    def test_missing_key_shown_as_not_configured(self):
        assert env_loader.describe()["api_key"] == "(未配置)"

    def test_describe_reports_all_fields(self, monkeypatch):
        monkeypatch.setenv(env_loader.ENV_API_KEY, "k")
        info = env_loader.describe()
        for field in ("env_file", "env_file_exists", "api_key", "api_key_env",
                      "base_url", "model_name", "judge_model_name", "missing"):
            assert field in info


# ============================================================
# CLI 集成
# ============================================================
class TestCliConfigCommand:

    def test_config_command_prints_and_exits_zero(self, monkeypatch, capsys):
        from eval.cli import main as cli_main
        monkeypatch.setenv(env_loader.ENV_API_KEY, "k-test-key-value-1234")
        monkeypatch.setenv(env_loader.ENV_BASE_URL, "https://example.com/v1")
        monkeypatch.setenv(env_loader.ENV_MODEL_NAME, "some-model")

        code = cli_main.main(["config"])
        out = capsys.readouterr().out
        assert code == cli_main.EXIT_OK
        assert "some-model" in out
        assert "https://example.com/v1" in out
        # 完整密钥不能出现在终端输出里
        assert "k-test-key-value-1234" not in out

    def test_config_command_exit_code_when_missing(self, monkeypatch, capsys):
        from eval.cli import main as cli_main
        monkeypatch.delenv(env_loader.ENV_API_KEY, raising=False)
        monkeypatch.delenv(env_loader.LEGACY_API_KEY, raising=False)
        monkeypatch.setattr(env_loader, "ENV_FILE",
                            os.path.join(PROJECT_ROOT, "definitely_absent.env"))

        code = cli_main.main(["config"])
        capsys.readouterr()
        assert code == cli_main.EXIT_CONFIG_ERROR

    def test_check_config_reports_missing(self, monkeypatch):
        from eval.cli import main as cli_main
        monkeypatch.delenv(env_loader.ENV_API_KEY, raising=False)
        monkeypatch.delenv(env_loader.LEGACY_API_KEY, raising=False)
        monkeypatch.setattr(env_loader, "ENV_FILE", os.devnull)
        assert env_loader.ENV_API_KEY in cli_main.check_config()

    def test_check_config_skipped_when_not_required(self):
        from eval.cli import main as cli_main
        assert cli_main.check_config(require_api=False) == []

    def test_config_error_message_points_to_env_file(self, capsys):
        from eval.cli import main as cli_main
        cli_main.print_config_error([env_loader.ENV_API_KEY])
        err = capsys.readouterr().err
        assert ".env" in err
        assert "CONFIGURATION ERROR" in err