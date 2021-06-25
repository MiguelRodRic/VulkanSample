#pragma once

#include "vulkan/vulkan_core.h"
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0; //Index of the binding in the array of bindings
        bindingDescription.stride = sizeof(Vertex); //Number of bytes from one entry to the next
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; //Move to the next data after each vertex or after each instance

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0; //From which binding index the per-vertex data comes
        attributeDescriptions[0].location = 0; //References the location directive of the input in the vertex shader (location 0 = position)
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; //Format for vec3
        attributeDescriptions[0].offset = offsetof(Vertex, pos); //Byte size of that data

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1; //location 1 = color
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2; //location 2 = uv
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);


        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const
    {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

namespace std
{
    template<> struct hash<Vertex>
    {
        size_t operator()(Vertex const& vertex) const
        {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

struct QueueFamilyIndices
{
    std::pair<uint32_t, bool> graphicsFamily;
    std::pair<uint32_t, bool> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.second && presentFamily.second;
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    std::vector<VkSurfaceFormatKHR> surfaceFormats;		//Color depth
    std::vector<VkPresentModeKHR> surfacePresentModes;	//Conditions for swapping images
};

class VulkanApp
{

public:
    void InitWindow();

    void InitVulkan();

    void DrawFrame();

    void MainLoop();

    void Cleanup();

    void Run();

    VkInstance GetInstance() const { return m_Instance; }

    VkDebugUtilsMessengerEXT* GetDebugMessenger() { return &m_DebugMessenger; }

private:
    void CreateInstance();

    void CreateSyncObjects();

    void CreateCommandBuffers();

    void CreateDescriptorSets();
    void CreateDescriptorPool();

    void CreateUniformBuffers();
    void CreateIndexBuffer();
    void CreateVertexBuffer();
    void CreateTextureSampler();
    void CreateTextureImageView();
    void CreateTextureImage();
    void CreateSurface();

    void CreateDepthResources();
    VkFormat FindDepthFormat();

    void CreateColorResources();

    bool HasStencilComponent(VkFormat format);

    VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

    void GenerateMipMaps(VkImage image, VkFormat imageFormat, uint32_t texWidth, int32_t texHeight, uint32_t mipLevels); 

    void CreateImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
        VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    
    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice aDevice);
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice aPhysicalDevice);
    VkSurfaceFormatKHR ChooseSwapChainSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& aSomeAvailableFormats);
    VkPresentModeKHR ChooseSwapChainPresentMode(const std::vector<VkPresentModeKHR> aSomeAvailablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& aCapabilities);
    
    bool IsPhysicalDeviceSuitable(VkPhysicalDevice aPhysicalDevice);

    bool CheckDeviceExtensionSupport(VkPhysicalDevice aPhysicalDevice);


    VkSampleCountFlagBits GetMaxUsableSampleCount();

    void PickPhysicalDevice();

    void CreateLogicalDevice();
    
    bool CheckValidationLayerSupport();

    std::vector<const char*> GetRequiredExtensions();
    
    VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t miplevels);
    void CreateImageViews();

    void CreateSwapChain();
    void CreateRenderPass();
    void CreateGraphicsPipeline();
    void CreateFrameBuffers();
    void CreateCommandPool();
    void CreateDescriptorSetLayout();

    void CleanupSwapChain();
    void RecreateSwapChain();
    
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer commandBuffer);
        
    void LoadModel();

    void UpdateUniformBuffer(uint32_t currentImage);
    
private:

    GLFWwindow* m_Window;

    VkInstance m_Instance;

    VkDebugUtilsMessengerEXT m_DebugMessenger;

    VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;

    VkDevice m_LogicalDevice;

    VkQueue m_GraphicsQueue;

    VkSurfaceKHR m_Surface;

    VkQueue m_PresentQueue;

    VkSwapchainKHR m_SwapChain;

    std::vector<VkImage> m_SwapChainImages;

    VkFormat m_SwapChainImageFormat;

    VkExtent2D m_SwapChainExtent;

    std::vector<VkImageView> m_SwapChainImageViews;

    VkRenderPass m_RenderPass;

    VkDescriptorSetLayout m_DescriptorSetLayout;

    VkDescriptorPool m_DescriptorPool;

    std::vector<VkDescriptorSet> m_DescriptorSets;

    VkPipelineLayout m_PipelineLayout;

    VkPipeline m_GraphicsPipeline;

    std::vector<VkFramebuffer> m_SwapChainFrameBuffers;

    VkCommandPool m_CommandPool;

    std::vector<Vertex> m_Vertices;

    std::vector<uint32_t> m_Indices;

    VkBuffer m_VertexBuffer;

    VkDeviceMemory m_VertexBufferMemory;

    VkBuffer m_IndexBuffer;

    VkDeviceMemory m_IndexBufferMemory;

    uint32_t m_MipLevels;

    VkImage m_TextureImage;

    VkDeviceMemory m_TextureImageMemory;

    VkImageView m_TextureImageView;

    VkSampler m_TextureSampler;

    VkImage m_DepthImage;

    VkDeviceMemory m_DepthImageMemory;

    VkImageView m_DepthImageView;

    VkImage m_ColorImage;

    VkDeviceMemory m_ColorImageMemory;

    VkImageView m_ColorImageView;

    std::vector<VkBuffer> m_UniformBuffers;

    std::vector<VkDeviceMemory> m_UniformBuffersMemory;

    std::vector<VkCommandBuffer> m_CommandBuffers;

    std::vector<VkSemaphore> m_ImageAvailableSemaphores;

    std::vector<VkSemaphore> m_RenderFinishedSemaphores;

    std::vector<VkFence> m_InFlightFences;

    size_t m_CurrentFrame = 0;

    VkSampleCountFlagBits m_MsaaSamples = VK_SAMPLE_COUNT_1_BIT; //No multisampling by default
};


namespace Debug
{
    VkResult CreateDebugUtilsMessengerEXT(VkInstance anInstance, const VkDebugUtilsMessengerCreateInfoEXT* apCreateInfo,
        const VkAllocationCallbacks* apAllocator, VkDebugUtilsMessengerEXT* apDebugMessenger);

    void DestroyDebugUtilsMessengerEXT(VkInstance anInstance, VkDebugUtilsMessengerEXT aDebugMessenger, const VkAllocationCallbacks* apAllocator);


    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& aCreateInfo);

    void SetupDebugMessenger(VulkanApp app);


    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallBack(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
}

