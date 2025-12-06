from click.testing import CliRunner
import pytest
from cli.cli import cli
import os
from PIL import Image

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_predict(runner, tmp_path):
    img_path = tmp_path / "input.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(img_path, format="JPEG")
    
    result = runner.invoke(cli, ['predict', str(img_path)], catch_exceptions=False)
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    assert "The predicted class" in result.output

def test_cli_resize(runner, tmp_path):
    img_path = tmp_path / "input.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(img_path, format="JPEG")
    
    output_path = tmp_path / "resized.jpg"
    
    result = runner.invoke(cli, ['resize', str(img_path), str(output_path), '--width', '50', '--height', '50'])
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    
    assert output_path.exists()
    
    with Image.open(output_path) as img:
        assert img.size == (50, 50)

def test_cli_grayscale(runner, tmp_path):
    img_path = tmp_path / "input.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(img_path, format="JPEG")
    
    output_path = tmp_path / "gray.jpg"
    
    result = runner.invoke(cli, ['grayscale', str(img_path), str(output_path)])
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    
    assert output_path.exists()
    
    with Image.open(output_path) as img:
        assert img.mode == 'L'
