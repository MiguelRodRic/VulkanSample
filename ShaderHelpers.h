#pragma once
#include <fstream>
#include <vector>

namespace ShaderHelpers
{
	std::vector<char> ReadFile(const std::string& aFilename)
	{
		std::ifstream file(aFilename, std::ios::ate | std::ios::binary);
		//We use ios::ate option to start reading at the end of file, therefore knowing the size of the file for the buffer allocation

		if (!file.is_open())
			throw std::runtime_error("Could not open file");

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	VkShaderModule CreateShaderModule(VkDevice aDevice, const std::vector<char>& aSomeShaderCode)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = aSomeShaderCode.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(aSomeShaderCode.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(aDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("failed to create a shader module");

		return shaderModule;
	}
}